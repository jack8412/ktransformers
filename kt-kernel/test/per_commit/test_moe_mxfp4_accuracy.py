#!/usr/bin/env python
# coding=utf-8
"""MXFP4 MoE accuracy + native-checkpoint loader tests for KT-Kernel x86 backends.

Covers:
  - bitwise ue8m0 -> bf16 scale conversion (loader) against a torch reference
  - the E2M1 -> BF16 PSHUFB LUT byte tables (mirrors operators/amx/fp4-moe.hpp)
  - MoE forward accuracy vs a PyTorch reference built from the exact dequantized
    weights, with true power-of-two (OCP MX) group scales, across the
    mat-vec / mat-mat dispatch boundary (qlen > 4 * expert_num / top_k)
  - MXFP4SafeTensorLoader resolution of both supported checkpoint namings:
    DeepSeek-V4-Flash ({base}.ffn.experts.{i}.{w1,w3,w2}.{weight,scale}) and
    Kimi-K3 compressed-tensors
    ({base}.block_sparse_moe.experts.{i}.{w1,w3,w2}.{weight_packed,weight_scale})
  - bitwise output equivalence: V4 naming vs K3 naming vs flat-pointer load
  - NativeMoEWrapper end-to-end on a synthetic K3-named checkpoint dir
  - the group_size == 32 guard
"""

import importlib.util
import os
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

register_cpu_ci(est_time=180, suite="default")

import pytest
import torch
import kt_kernel_ext

KT_KERNEL_ROOT = Path(__file__).resolve().parents[2]

expert_num = 8
hidden_size = 256
intermediate_size = 512
num_experts_per_tok = 2
max_len = 512
group_size = 32
validation_iter = 3
CPUINFER_PARAM = 16

# E2M1 value table, indexed by the 4-bit code (bit 3 = sign).
E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

# BF16 low/high byte LUTs for the 16 E2M1 codes. Must stay in sync with
# fp4_bf16_lo / fp4_bf16_hi in operators/amx/fp4-moe.hpp (and the identical
# tables in operators/avx2/mxfp4-moe.hpp).
FP4_BF16_LO = [0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0, 0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0]
FP4_BF16_HI = [0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40, 0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0]


def load_amx_utils():
    pkg_root = KT_KERNEL_ROOT / "python"
    utils_root = pkg_root / "utils"

    if "kt_kernel" not in sys.modules:
        kt_kernel_pkg = types.ModuleType("kt_kernel")
        kt_kernel_pkg.__path__ = [str(pkg_root)]
        kt_kernel_pkg.kt_kernel_ext = kt_kernel_ext
        sys.modules["kt_kernel"] = kt_kernel_pkg

    if "kt_kernel_ext" not in sys.modules:
        sys.modules["kt_kernel_ext"] = kt_kernel_ext

    if "kt_kernel.utils" not in sys.modules:
        utils_pkg = types.ModuleType("kt_kernel.utils")
        utils_pkg.__path__ = [str(utils_root)]
        sys.modules["kt_kernel.utils"] = utils_pkg

    module_specs = [
        ("kt_kernel.experts_base", pkg_root / "experts_base.py"),
        ("kt_kernel.utils.loader", utils_root / "loader.py"),
        ("kt_kernel.utils.amx", utils_root / "amx.py"),
    ]
    for module_name, module_path in module_specs:
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        assert spec.loader is not None
        spec.loader.exec_module(module)

    return sys.modules["kt_kernel.utils.amx"]


def load_loader_module():
    load_amx_utils()
    return sys.modules["kt_kernel.utils.loader"]


def ue8m0_to_bf16_ref(scale_u8):
    """Reference ue8m0 -> bf16: 2^(e-127) for e in [1, 254]; e=0 -> +0.0
    (loader convention; 2^-127 is below the bf16 normal range), e=255 -> +inf."""
    e = scale_u8.to(torch.int32)
    vals = torch.ldexp(torch.ones_like(e, dtype=torch.float32), e - 127)
    vals = torch.where(e == 0, torch.zeros_like(vals), vals)
    vals = torch.where(e == 255, torch.full_like(vals, float("inf")), vals)
    return vals.to(torch.bfloat16)


def make_mxfp4_weight(n, k, gen, exp_lo=-9, exp_hi=-4):
    """Directly generate a random MXFP4 weight: E2M1 codes + power-of-two scales.

    Returns (packed uint8 [n, k/2], scale_u8 uint8 [n, k/32] (ue8m0),
             scale_bf16 [n, k/32], dequant fp32 [n, k]).
    """
    assert k % group_size == 0 and k % 2 == 0
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.int64)
    packed = ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8).contiguous()
    scale_u8 = torch.randint(127 + exp_lo, 127 + exp_hi + 1, (n, k // group_size), generator=gen, dtype=torch.int64).to(
        torch.uint8
    )
    scale_bf16 = ue8m0_to_bf16_ref(scale_u8).contiguous()
    scale_f32 = scale_bf16.float()
    dequant = E2M1_VALUES[codes] * scale_f32.repeat_interleave(group_size, dim=1)
    return packed, scale_u8.contiguous(), scale_bf16, dequant


def make_mxfp4_experts(seed, hidden=hidden_size, inter=intermediate_size):
    """Per-expert MXFP4 gate/up/down weights at [inter, hidden] / [hidden, inter]."""
    gen = torch.Generator().manual_seed(seed)
    data = {"packed": {}, "scale_u8": {}, "scale_bf16": {}, "dequant": {}}
    for proj, (n, k) in (("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))):
        packed, s_u8, s_bf16, deq = [], [], [], []
        for _ in range(expert_num):
            p, u, b, d = make_mxfp4_weight(n, k, gen)
            packed.append(p)
            s_u8.append(u)
            s_bf16.append(b)
            deq.append(d)
        data["packed"][proj] = packed
        data["scale_u8"][proj] = s_u8
        data["scale_bf16"][proj] = s_bf16
        data["dequant"][proj] = torch.stack(deq)
    return data


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def situ_act(gate, up, beta, linear_beta):
    """Reference SituAndMul (Kimi-K3 modeling_kimi_linear.py):
    beta*tanh(gate/beta)*sigmoid(gate) * up, with up squashed by linear_beta."""
    situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta:
        up = linear_beta * torch.tanh(up / linear_beta)
    return situ_a * up


def mlp_torch(input_data, gate_proj, up_proj, down_proj, situ=None):
    gate_buf = torch.mm(input_data, gate_proj.t())
    up_buf = torch.mm(input_data, up_proj.t())
    if situ is None:
        intermediate = act_fn(gate_buf) * up_buf
    else:
        # The kernel rounds the activation output to bf16 before the down GEMM.
        intermediate = situ_act(gate_buf, up_buf, *situ).to(torch.bfloat16).float()
    return torch.mm(intermediate, down_proj.t())


def moe_torch(input_data, expert_ids, weights, gate_proj, up_proj, down_proj, situ=None):
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    tokens_per_expert = cnts.sum(dim=0)
    idxs = expert_ids.view(-1).argsort()
    sorted_tokens = input_data[idxs // expert_ids.shape[1]]
    outputs = []
    start_idx = 0
    for i, num_tokens in enumerate(tokens_per_expert):
        end_idx = start_idx + num_tokens
        if num_tokens == 0:
            continue
        tokens = sorted_tokens[start_idx:end_idx]
        out = mlp_torch(tokens, gate_proj[i], up_proj[i], down_proj[i], situ=situ)
        outputs.append(out)
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*expert_ids.shape, -1).float().mul_(weights.unsqueeze(-1)).sum(1)).to(new_x.dtype)


def available_numa_nodes():
    try:
        return sorted(
            int(d[len("node") :])
            for d in os.listdir("/sys/devices/system/node")
            if d.startswith("node") and d[len("node") :].isdigit()
        )
    except OSError:
        return [0]


_cpu_infer_cache = {}


def make_cpu_infer(threads=CPUINFER_PARAM):
    """CPUInfer with an explicit worker-pool config: up to 2 subpools on real
    NUMA nodes, so tests behave identically on 1-, 2- and 6-node hosts
    (CPUInfer(int) auto-derives one subpool per node, which breaks
    intermediate_size % tp_count on many-node machines)."""
    if threads in _cpu_infer_cache:
        return _cpu_infer_cache[threads]
    nodes = available_numa_nodes()
    count = 2 if len(nodes) >= 2 else 1
    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = count
    wp.subpool_numa_map = nodes[:count]
    wp.subpool_thread_count = [max(1, threads // count)] * count
    cpu_infer = kt_kernel_ext.CPUInfer(wp)
    _cpu_infer_cache[threads] = cpu_infer
    return cpu_infer


def available_backends():
    backends = []
    if hasattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE"):
        backends.append(("AMXFP4_KGroup_MOE", kt_kernel_ext.moe.AMXFP4_KGroup_MOE, 0.05))
    if hasattr(kt_kernel_ext.moe, "AVX2MXFP4_MOE"):
        backends.append(("AVX2MXFP4_MOE", kt_kernel_ext.moe.AVX2MXFP4_MOE, 0.05))
    return backends


def build_flat_config(data, hidden=hidden_size, inter=intermediate_size, qlen_max=max_len):
    """Flat-pointer MOEConfig. Returns (config, keepalive) — keepalive tensors
    must stay referenced while the backend is in use."""
    gate_qw = torch.stack(data["packed"]["gate"]).contiguous()
    up_qw = torch.stack(data["packed"]["up"]).contiguous()
    down_qw = torch.stack(data["packed"]["down"]).contiguous()
    gate_sc = torch.stack(data["scale_u8"]["gate"]).contiguous()
    up_sc = torch.stack(data["scale_u8"]["up"]).contiguous()
    down_sc = torch.stack(data["scale_u8"]["down"]).contiguous()

    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden, inter, 0)
    config.max_len = qlen_max
    config.gate_proj = gate_qw.data_ptr()
    config.up_proj = up_qw.data_ptr()
    config.down_proj = down_qw.data_ptr()
    config.gate_scale = gate_sc.data_ptr()
    config.up_scale = up_sc.data_ptr()
    config.down_scale = down_sc.data_ptr()
    config.quant_config.bits = 4
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    keepalive = (gate_qw, up_qw, down_qw, gate_sc, up_sc, down_sc)
    return config, keepalive


def build_per_expert_config(weights_dict, hidden=hidden_size, inter=intermediate_size, qlen_max=max_len):
    """Per-expert-pointer MOEConfig from a loader-style dict (uint8 weights,
    bf16 scales, one tensor per expert)."""
    config = kt_kernel_ext.moe.MOEConfig(expert_num, num_experts_per_tok, hidden, inter, 0)
    config.max_len = qlen_max
    config.gate_projs = [[t.data_ptr() for t in weights_dict["gate"]]]
    config.up_projs = [[t.data_ptr() for t in weights_dict["up"]]]
    config.down_projs = [[t.data_ptr() for t in weights_dict["down"]]]
    config.gate_scales = [[t.data_ptr() for t in weights_dict["gate_scale"]]]
    config.up_scales = [[t.data_ptr() for t in weights_dict["up_scale"]]]
    config.down_scales = [[t.data_ptr() for t in weights_dict["down_scale"]]]
    config.quant_config.bits = 4
    config.quant_config.group_size = group_size
    config.quant_config.zero_point = False
    return config


def load_and_forward(backend_cls, config, cpu_infer, qlen, expert_ids, weights, input_data, hidden=hidden_size):
    physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
    moe = backend_cls(config)
    cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
    cpu_infer.sync()

    output = torch.empty((qlen, hidden), dtype=torch.bfloat16).contiguous()
    bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            bsz_tensor.data_ptr(),
            num_experts_per_tok,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            input_data.data_ptr(),
            output.data_ptr(),
            False,
        )
    )
    cpu_infer.sync()
    return output


def make_routing(qlen, gen=None):
    expert_ids = torch.stack([torch.randperm(expert_num, generator=gen)[:num_experts_per_tok] for _ in range(qlen)]).contiguous()
    weights = torch.rand((qlen, num_experts_per_tok), dtype=torch.float32, generator=gen).contiguous()
    return expert_ids, weights


# ---------------------------------------------------------------------------
# Bitwise decode building blocks
# ---------------------------------------------------------------------------


def test_ue8m0_to_bf16_bitwise():
    loader_mod = load_loader_module()
    all_u8 = torch.arange(256, dtype=torch.uint8)
    got = loader_mod.MXFP4SafeTensorLoader._ue8m0_to_bf16(all_u8)
    ref = ue8m0_to_bf16_ref(all_u8)
    assert got.dtype == torch.bfloat16
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16)), "ue8m0->bf16 must be bit-exact for all 256 codes"


def test_mxfp4_resident_footprint():
    """MXFP4 keeps E8M0 scales resident as 1 B/group, so a weight matrix costs
    0.5 + 1/32 = 0.53125 B/elem instead of 0.625 with fp32 scales (-15%)."""
    if not hasattr(kt_kernel_ext.moe, "mxfp4_buffer_bytes"):
        pytest.skip("mxfp4_buffer_bytes not available (non-AVX512 build)")

    K3_HIDDEN, K3_INTER = 3584, 3072  # Kimi-K3 routed-expert dims
    for n, k in ((K3_INTER, K3_HIDDEN), (K3_HIDDEN, K3_INTER)):
        got = kt_kernel_ext.moe.mxfp4_buffer_bytes(n, k, group_size)
        expect = n * k // 2 + n * (k // group_size)  # both terms 64B-aligned here
        assert got == expect, f"{n}x{k}: {got} != {expect}"
        per_elem = got / (n * k)
        assert abs(per_elem - 0.53125) < 1e-6, f"{n}x{k}: {per_elem} B/elem"

    # Projected full-model expert footprint: 92 MoE layers x 896 experts.
    params = 92 * 896 * (2 * K3_INTER * K3_HIDDEN + K3_HIDDEN * K3_INTER)
    resident, prev = params * 0.53125, params * 0.625
    print(
        f"  K3 expert params={params/1e12:.2f}e12  resident={resident/1e12:.3f} TB "
        f"(was {prev/1e12:.3f} TB, saves {(prev-resident)/1e9:.0f} GB)"
    )
    assert params == 2722740830208


def test_e2m1_lut_tables_match_spec():
    lut_bits = torch.tensor([(hi << 8) | lo for lo, hi in zip(FP4_BF16_LO, FP4_BF16_HI)], dtype=torch.int32)
    lut_vals = lut_bits.to(torch.int16).view(torch.bfloat16)
    ref = E2M1_VALUES.to(torch.bfloat16)
    assert torch.equal(lut_vals.view(torch.int16), ref.view(torch.int16)), (
        "fp4_bf16_lo/hi byte tables must decode to the E2M1 value set bit-exactly "
        "(including -0.0 at code 8)"
    )


# ---------------------------------------------------------------------------
# Kernel accuracy vs dequantized-weight torch reference
# ---------------------------------------------------------------------------


def run_backend_accuracy_test(
    backend_name, backend_cls, threshold, qlen, hidden=hidden_size, inter=intermediate_size, situ=None
):
    cpu_infer = make_cpu_infer()
    with torch.inference_mode():
        data = make_mxfp4_experts(seed=42 + qlen, hidden=hidden, inter=inter)
        config, keepalive = build_flat_config(data, hidden=hidden, inter=inter, qlen_max=max(max_len, qlen))
        config.pool = cpu_infer.backend_
        if situ is not None:
            config.situ_beta, config.situ_linear_beta = situ[0], (situ[1] or 0.0)

        gen = torch.Generator().manual_seed(1234 + qlen)
        print(f"\n--- {backend_name} (qlen={qlen}, hidden={hidden}, inter={inter}) ---")
        physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
        moe = backend_cls(config)
        cpu_infer.submit(moe.load_weights_task(physical_to_logical_map.data_ptr()))
        cpu_infer.sync()

        for i in range(validation_iter):
            expert_ids, weights = make_routing(qlen, gen)
            input_data = (torch.randn((qlen, hidden), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden), dtype=torch.bfloat16).contiguous()
            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
            cpu_infer.submit(
                moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            cpu_infer.sync()

            ref_output = moe_torch(
                input_data.float(),
                expert_ids,
                weights,
                data["dequant"]["gate"],
                data["dequant"]["up"],
                data["dequant"]["down"],
                situ=situ,
            ).to(torch.bfloat16)
            diff = torch.mean(torch.abs(output.float() - ref_output.float())) / (
                torch.mean(torch.abs(ref_output.float())) + 1e-8
            )
            print(f"  Iteration {i}: diff = {diff.item():.6f}")
            assert diff < threshold, f"{backend_name} accuracy failed: diff={diff.item():.6f} >= {threshold}"
    del keepalive


def test_mxfp4_accuracy():
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend (AMXFP4_KGroup_MOE / AVX2MXFP4_MOE) available")

    # qlen 1 and 8 take the mat-vec path, 32 crosses the
    # qlen > 4 * expert_num / num_experts_per_tok (= 16) mat-mat dispatch.
    for backend_name, backend_cls, threshold in backends:
        for qlen in (1, 8, 32):
            run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=qlen)
        # K/N sizes that are multiples of 32 but not of 64/128/256, to cross
        # k-group and N blocking edges (inter=192 stays 32-aligned after the
        # 2-way NUMA TP split used by make_cpu_infer).
        run_backend_accuracy_test(backend_name, backend_cls, threshold, qlen=32, hidden=160, inter=192)


def test_mxfp4_situ_accuracy():
    """Kimi-K3 'situ' activation: beta*tanh(g/beta)*sigmoid(g) * linear_beta*tanh(u/linear_beta)."""
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend available")

    # K3: activation_situ_beta=4.0, activation_situ_linear_beta=25.0.
    for backend_name, backend_cls, threshold in backends:
        for situ in ((4.0, 25.0), (4.0, None), (1.0, 25.0)):
            for qlen in (1, 32):
                run_backend_accuracy_test(
                    f"{backend_name} situ{situ}", backend_cls, threshold, qlen=qlen, situ=situ
                )


def test_situ_differs_from_silu():
    """Guard against the config field being silently ignored: with the same
    weights, situ output must differ from the silu output, and must match its
    own reference (which the accuracy test above checks)."""
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend available")
    _, backend_cls, _ = backends[0]

    cpu_infer = make_cpu_infer()
    gen = torch.Generator().manual_seed(21)
    qlen = 8
    expert_ids, weights = make_routing(qlen, gen)
    input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
    data = make_mxfp4_experts(seed=21)

    outs = {}
    for tag, situ in (("silu", None), ("situ", (4.0, 25.0))):
        config, keepalive = build_flat_config(data)
        config.pool = cpu_infer.backend_
        if situ is not None:
            config.situ_beta, config.situ_linear_beta = situ
        outs[tag] = load_and_forward(backend_cls, config, cpu_infer, qlen, expert_ids, weights, input_data)
        del keepalive

    assert torch.isfinite(outs["situ"].float()).all(), "situ produced non-finite output"
    assert not torch.equal(outs["silu"], outs["situ"]), "situ_beta had no effect — config field ignored?"


def test_situ_rejected_for_silu_only_backends():
    """LLAMAFILE / MOE_INT* have a hard-coded scalar silu epilogue that never
    reads situ_beta; the factory must refuse instead of serving wrong math."""
    load_amx_utils()
    import importlib.util

    pkg_root = KT_KERNEL_ROOT / "python"
    spec = importlib.util.spec_from_file_location("kt_kernel.experts", pkg_root / "experts.py")
    experts_mod = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel.experts"] = experts_mod
    spec.loader.exec_module(experts_mod)

    common = dict(
        layer_idx=0,
        num_experts=expert_num,
        num_experts_per_tok=num_experts_per_tok,
        hidden_size=hidden_size,
        moe_intermediate_size=intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=CPUINFER_PARAM,
        threadpool_count=1,
        weight_path="/nonexistent",
        chunked_prefill_size=max_len,
    )
    # LLAMAFILE / MOE_INT* hard-code scalar silu; SYCL_GPTQ_INT4 routes to
    # NativeMoEWrapper but fuses its activation into device kernels that never
    # read situ_beta (the base skips apply_activation for it entirely).
    for bad_method in ("LLAMAFILE", "MOE_INT4", "MOE_INT8", "SYCL_GPTQ_INT4"):
        with pytest.raises(ValueError, match="situ_beta"):
            experts_mod.KTMoEWrapper(method=bad_method, situ_beta=4.0, situ_linear_beta=25.0, **common)
    # Every method the allow-list claims to support must actually be listed as
    # an inference method (guards against a typo silently disabling situ).
    assert experts_mod.SITU_SUPPORTED_METHODS <= experts_mod.INFERENCE_METHODS
    assert "SYCL_GPTQ_INT4" not in experts_mod.SITU_SUPPORTED_METHODS
    # linear_beta without beta is a config error, not a silent no-op.
    with pytest.raises(ValueError, match="requires situ_beta"):
        experts_mod.KTMoEWrapper(method="MXFP4", situ_beta=0.0, situ_linear_beta=25.0, **common)
    # situ + swiglu clamp would silently drop the clamp.
    with pytest.raises(ValueError, match="cannot be combined"):
        experts_mod.KTMoEWrapper(method="MXFP4", situ_beta=4.0, swiglu_limit=10.0, **common)
    # SFT backends have their own epilogue.
    with pytest.raises(ValueError, match="situ_beta"):
        experts_mod.KTMoEWrapper(method="AMXBF16_SFT", mode="sft", situ_beta=4.0, **common)


# ---------------------------------------------------------------------------
# Native-checkpoint loader: V4 + K3 namings
# ---------------------------------------------------------------------------


def _save_checkpoint(dirname, data, naming):
    """Write a synthetic single-layer expert checkpoint in the given naming."""
    from safetensors.torch import save_file

    tensors = {}
    proj_map = {"w1": "gate", "w3": "up", "w2": "down"}
    for e in range(expert_num):
        for wname, proj in proj_map.items():
            packed = data["packed"][proj][e].clone()
            scale = data["scale_u8"][proj][e].clone()
            if naming == "v4":
                base = f"layers.0.ffn.experts.{e}.{wname}"
                tensors[f"{base}.weight"] = packed
                tensors[f"{base}.scale"] = scale
            elif naming == "k3":
                base = f"language_model.model.layers.1.block_sparse_moe.experts.{e}.{wname}"
                tensors[f"{base}.weight_packed"] = packed
                tensors[f"{base}.weight_scale"] = scale
            else:
                raise ValueError(naming)
    save_file(tensors, os.path.join(dirname, "model.safetensors"))


def _assert_loader_dict_matches(weights_dict, data):
    for proj in ("gate", "up", "down"):
        assert len(weights_dict[proj]) == expert_num
        for e in range(expert_num):
            assert weights_dict[proj][e].dtype == torch.uint8
            assert torch.equal(weights_dict[proj][e], data["packed"][proj][e])
            got_scale = weights_dict[f"{proj}_scale"][e]
            assert got_scale.dtype == torch.uint8, "MXFP4 scales stay resident as raw ue8m0 bytes"
            assert torch.equal(got_scale, data["scale_u8"][proj][e])


def test_k3_naming_loader():
    loader_mod = load_loader_module()
    data = make_mxfp4_experts(seed=7)
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, data, naming="k3")
        loader = loader_mod.MXFP4SafeTensorLoader(tmpdir)
        # Direct base key (what NativeMoEWrapper probes as its 2nd candidate)
        weights = loader.load_experts("language_model.model.layers.1")
        _assert_loader_dict_matches(weights, data)
        # The V4 naming must NOT match a K3-named checkpoint under other bases
        with pytest.raises(ValueError, match="No MXFP4 experts"):
            loader.load_experts("model.layers.1")
        loader.close_all_handles()


def test_v4_naming_loader():
    loader_mod = load_loader_module()
    data = make_mxfp4_experts(seed=8)
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, data, naming="v4")
        loader = loader_mod.MXFP4SafeTensorLoader(tmpdir)
        weights = loader.load_experts("model.layers.0")
        _assert_loader_dict_matches(weights, data)
        loader.close_all_handles()


def test_v4_k3_flat_forward_equivalence():
    """The same weights loaded via V4 naming, K3 naming, and flat pointers must
    produce bitwise-identical forward outputs."""
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend available")
    backend_name, backend_cls, _ = backends[0]
    loader_mod = load_loader_module()

    data = make_mxfp4_experts(seed=9)
    cpu_infer = make_cpu_infer()
    gen = torch.Generator().manual_seed(99)
    qlen = 8
    expert_ids, weights = make_routing(qlen, gen)
    input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()

    outputs = {}
    with tempfile.TemporaryDirectory() as tmp_v4, tempfile.TemporaryDirectory() as tmp_k3:
        _save_checkpoint(tmp_v4, data, naming="v4")
        _save_checkpoint(tmp_k3, data, naming="k3")
        for tag, tmpdir, base in (("v4", tmp_v4, "model.layers.0"), ("k3", tmp_k3, "language_model.model.layers.1")):
            loader = loader_mod.MXFP4SafeTensorLoader(tmpdir)
            wdict = loader.load_experts(base)
            config = build_per_expert_config(wdict)
            config.pool = cpu_infer.backend_
            outputs[tag] = load_and_forward(backend_cls, config, cpu_infer, qlen, expert_ids, weights, input_data)
            loader.close_all_handles()

        config, keepalive = build_flat_config(data)
        config.pool = cpu_infer.backend_
        outputs["flat"] = load_and_forward(backend_cls, config, cpu_infer, qlen, expert_ids, weights, input_data)
        del keepalive

    assert torch.isfinite(outputs["flat"].float()).all()
    assert outputs["flat"].abs().sum() > 0
    assert torch.equal(outputs["v4"], outputs["k3"]), f"{backend_name}: V4 vs K3 naming outputs differ"
    assert torch.equal(outputs["v4"], outputs["flat"]), f"{backend_name}: per-expert vs flat load outputs differ"


# ---------------------------------------------------------------------------
# NativeMoEWrapper end-to-end on a K3-named checkpoint
# ---------------------------------------------------------------------------


def _can_pin_memory():
    try:
        torch.empty(1, pin_memory=True)
        return True
    except RuntimeError:
        return False


def test_native_wrapper_e2e_k3():
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend available")
    if not _can_pin_memory():
        pytest.skip("pinned memory unavailable (CPU-only torch build)")

    amx_mod = load_amx_utils()
    data = make_mxfp4_experts(seed=11)
    with tempfile.TemporaryDirectory() as tmpdir:
        _save_checkpoint(tmpdir, data, naming="k3")
        amx_mod.NativeMoEWrapper.force_release_loader()
        wrapper = amx_mod.NativeMoEWrapper(
            layer_idx=1,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=CPUINFER_PARAM,
            threadpool_count=1,
            weight_path=tmpdir,
            chunked_prefill_size=max_len,
            method="MXFP4",
        )
        physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
        wrapper.load_weights(physical_to_logical_map)

        gen = torch.Generator().manual_seed(5)
        for qlen in (1, 32):
            expert_ids, weights = make_routing(qlen, gen)
            input_data = (torch.randn((qlen, hidden_size), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
            output = torch.empty((qlen, hidden_size), dtype=torch.bfloat16).contiguous()
            bsz_tensor = torch.tensor([qlen], dtype=torch.int32)
            wrapper.cpu_infer.submit(
                wrapper.moe.forward_task(
                    bsz_tensor.data_ptr(),
                    num_experts_per_tok,
                    expert_ids.data_ptr(),
                    weights.data_ptr(),
                    input_data.data_ptr(),
                    output.data_ptr(),
                    False,
                )
            )
            wrapper.cpu_infer.sync()

            ref_output = moe_torch(
                input_data.float(), expert_ids, weights, data["dequant"]["gate"], data["dequant"]["up"], data["dequant"]["down"]
            ).to(torch.bfloat16)
            diff = torch.mean(torch.abs(output.float() - ref_output.float())) / (
                torch.mean(torch.abs(ref_output.float())) + 1e-8
            )
            print(f"  NativeMoEWrapper e2e (K3 naming, qlen={qlen}): diff = {diff.item():.6f}")
            assert diff < 0.05, f"NativeMoEWrapper K3 e2e failed: diff={diff.item():.6f}"


def test_mxfp4_group_size_guard():
    """A checkpoint whose scales imply group_size != 32 must be rejected."""
    backends = available_backends()
    if not backends:
        pytest.skip("no MXFP4 backend available")
    if not _can_pin_memory():
        pytest.skip("pinned memory unavailable (CPU-only torch build)")

    from safetensors.torch import save_file

    amx_mod = load_amx_utils()
    gen = torch.Generator().manual_seed(13)
    bad_group = 16
    tensors = {}
    for e in range(expert_num):
        for wname, (n, k) in (("w1", (intermediate_size, hidden_size)), ("w3", (intermediate_size, hidden_size)), ("w2", (hidden_size, intermediate_size))):
            codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.int64)
            packed = ((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8)
            scale = torch.full((n, k // bad_group), 121, dtype=torch.uint8)
            base = f"language_model.model.layers.1.block_sparse_moe.experts.{e}.{wname}"
            tensors[f"{base}.weight_packed"] = packed.contiguous()
            tensors[f"{base}.weight_scale"] = scale.contiguous()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_file(tensors, os.path.join(tmpdir, "model.safetensors"))
        amx_mod.NativeMoEWrapper.force_release_loader()
        wrapper = amx_mod.NativeMoEWrapper(
            layer_idx=1,
            num_experts=expert_num,
            num_experts_per_tok=num_experts_per_tok,
            hidden_size=hidden_size,
            moe_intermediate_size=intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=CPUINFER_PARAM,
            threadpool_count=1,
            weight_path=tmpdir,
            chunked_prefill_size=max_len,
            method="MXFP4",
        )
        physical_to_logical_map = torch.tensor(range(expert_num), dtype=torch.int64).contiguous()
        with pytest.raises(ValueError, match="group_size == 32"):
            wrapper.load_weights(physical_to_logical_map)


# ---------------------------------------------------------------------------
# Direct runner (CI executes this file as `python3 <file>`)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_ue8m0_to_bf16_bitwise,
        test_mxfp4_resident_footprint,
        test_e2m1_lut_tables_match_spec,
        test_mxfp4_accuracy,
        test_mxfp4_situ_accuracy,
        test_situ_differs_from_silu,
        test_situ_rejected_for_silu_only_backends,
        test_k3_naming_loader,
        test_v4_naming_loader,
        test_v4_k3_flat_forward_equivalence,
        test_native_wrapper_e2e_k3,
        test_mxfp4_group_size_guard,
    ]
    failed = []
    for fn in tests:
        name = fn.__name__
        try:
            fn()
            print(f"[PASS] {name}")
        except pytest.skip.Exception as e:
            print(f"[SKIP] {name}: {e}")
        except Exception as e:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            print(f"[FAIL] {name}: {e}")
            failed.append(name)
    # Tear the worker pools down while the interpreter is alive: leaving two
    # live CPUInfer instances (this module's cache + the NativeMoEWrapper
    # singleton from the e2e tests) to static destructors aborts on exit.
    import gc

    _cpu_infer_cache.clear()
    base_mod = sys.modules.get("kt_kernel.experts_base")
    if base_mod is not None:
        base_mod._MoEBase._cpu_infer_instance = None
    gc.collect()

    if failed:
        print(f"\n{len(failed)} test(s) failed: {failed}")
        sys.exit(1)
    print(f"\nAll {len(tests)} MXFP4 tests passed (or skipped cleanly).")
