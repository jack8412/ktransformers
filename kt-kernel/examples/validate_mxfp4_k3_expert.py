#!/usr/bin/env python
# coding=utf-8
"""Validate the MXFP4 path against real Kimi-K3 checkpoint tensors.

Takes a directory (or single .safetensors file) containing at least one
expert's `language_model.model.layers.{L}.block_sparse_moe.experts.{E}.{w1,w3,w2}
.weight_packed/.weight_scale` tensors — e.g. a full checkpoint, or a small
range-read extract — and checks:

  1. shapes/dtypes match the compressed-tensors "mxfp4-pack-quantized" layout,
  2. MXFP4SafeTensorLoader resolves the K3 naming and converts scales exactly,
  3. our E2M1+ue8m0 dequant agrees bitwise with the `compressed_tensors`
     library's unpack (if importable — authoritative nibble-order check),
  4. per-k-group structure: every 32-element group's max |code| hits the top
     of the E2M1 grid (minmax observer property of the real checkpoint),
  5. backend forward on the real expert vs the dequantized torch reference.

Usage:
    python examples/validate_mxfp4_k3_expert.py /path/to/ckpt_or_file [--layer 1 --expert 0]
"""
import argparse
import os
import sys

import torch

try:
    import kt_kernel_ext
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__) + "/../build")
    from kt_kernel import kt_kernel_ext

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)
GROUP_SIZE = 32


def load_loader_cls():
    import importlib.util
    import types

    pkg_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python")
    if "kt_kernel" not in sys.modules:
        pkg = types.ModuleType("kt_kernel")
        pkg.__path__ = [pkg_root]
        sys.modules["kt_kernel"] = pkg
    spec = importlib.util.spec_from_file_location("kt_kernel.utils.loader", os.path.join(pkg_root, "utils", "loader.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kt_kernel.utils.loader"] = mod
    spec.loader.exec_module(mod)
    return mod.MXFP4SafeTensorLoader


def dequant_ours(packed, scale_bf16):
    lo = (packed & 0x0F).to(torch.int64)
    hi = ((packed >> 4) & 0x0F).to(torch.int64)
    codes = torch.stack((lo, hi), dim=-1).view(packed.shape[0], -1)
    return E2M1_VALUES[codes] * scale_bf16.float().repeat_interleave(GROUP_SIZE, dim=1)


def check_compressed_tensors_reference(packed, scale_u8, ours):
    """Bitwise cross-check against the compressed-tensors library, if present."""
    try:
        # compressed-tensors >= 0.17
        from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8
    except ImportError:
        try:  # older layout
            from compressed_tensors.compressors.quantized_compressors.nvfp4_quantized import unpack_fp4_from_uint8
        except ImportError:
            print("  [3] compressed-tensors not importable here -> skipping library cross-check")
            return None
    m, half_n = packed.shape
    ref_vals = unpack_fp4_from_uint8(packed, m, half_n * 2, dtype=torch.float32)
    ref = ref_vals * torch.ldexp(torch.ones(1), scale_u8.to(torch.int32) - 127).float().repeat_interleave(
        GROUP_SIZE, dim=1
    )
    ok = torch.equal(ours, ref)
    print(f"  [3] compressed-tensors unpack cross-check: {'BITWISE MATCH' if ok else 'MISMATCH'}")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--expert", type=int, default=0)
    p.add_argument("--threads", type=int, default=32)
    args = p.parse_args()

    from safetensors import safe_open

    path = args.path
    files = []
    if os.path.isdir(path):
        for root, _, fs in os.walk(path):
            files += [os.path.join(root, f) for f in fs if f.endswith(".safetensors")]
    else:
        files = [path]

    base = f"language_model.model.layers.{args.layer}.block_sparse_moe.experts.{args.expert}"
    tensors = {}
    for f in files:
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                if k.startswith(base + "."):
                    tensors[k[len(base) + 1 :]] = h.get_tensor(k)
    if not tensors:
        print(f"No tensors under {base} in {path}")
        sys.exit(1)

    ok = True

    # [1] shapes/dtypes
    print("[1] tensor layout:")
    shapes = {}
    for name, t in sorted(tensors.items()):
        print(f"  {name}: dtype={t.dtype} shape={tuple(t.shape)}")
        shapes[name] = tuple(t.shape)
    for w in ("w1", "w3", "w2"):
        pw, sc = tensors[f"{w}.weight_packed"], tensors[f"{w}.weight_scale"]
        assert pw.dtype == torch.uint8 and sc.dtype == torch.uint8, f"{w}: expected uint8 tensors"
        n, half_k = pw.shape
        assert sc.shape == (n, half_k * 2 // GROUP_SIZE), f"{w}: scale shape mismatch"
    inter, hidden = shapes["w1.weight_packed"][0], shapes["w1.weight_packed"][1] * 2
    assert shapes["w2.weight_packed"] == (hidden, inter // 2), "w2 must be [hidden, inter/2]"
    print(f"  -> expert dims: hidden={hidden} inter={inter}  [OK]")

    # [2] loader resolution + exact scale conversion
    loader_cls = load_loader_cls()
    loader = loader_cls(path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path)))
    wdict = loader.load_experts(f"language_model.model.layers.{args.layer}")
    exp = args.expert
    assert torch.equal(wdict["gate"][exp], tensors["w1.weight_packed"])
    expect_scale = (tensors["w1.weight_scale"].to(torch.int32) << 7).to(torch.int16).view(torch.bfloat16)
    assert torch.equal(wdict["gate_scale"][exp].view(torch.int16), expect_scale.view(torch.int16))
    print("[2] MXFP4SafeTensorLoader: K3 naming resolved, scale conversion exact  [OK]")

    deq = {}
    for proj, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
        deq[proj] = dequant_ours(tensors[f"{w}.weight_packed"], wdict[f"{proj}_scale"][exp])

    # [3] library cross-check (nibble order ground truth)
    ct = check_compressed_tensors_reference(tensors["w1.weight_packed"], tensors["w1.weight_scale"], deq["gate"])
    if ct is False:
        ok = False

    # [4] minmax-observer group structure
    codes = torch.stack(
        (
            (tensors["w1.weight_packed"] & 0x0F).to(torch.int64),
            ((tensors["w1.weight_packed"] >> 4) & 0x0F).to(torch.int64),
        ),
        dim=-1,
    ).view(inter, hidden)
    mags = E2M1_VALUES.abs()[codes]
    group_max = mags.view(inter, hidden // GROUP_SIZE, GROUP_SIZE).amax(dim=-1)
    frac_top = (group_max >= 3.0).float().mean().item()
    print(f"[4] fraction of k-groups whose max |code| >= 3.0: {frac_top:.4f} (minmax observer -> expect ~1.0)")
    if frac_top < 0.9:
        print("    WARNING: group structure unexpected — check group axis / nibble order")
        ok = False

    # [5] backend forward vs dequant reference
    name = "AMXFP4_KGroup_MOE" if hasattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE") else "AVX2MXFP4_MOE"
    cls = getattr(kt_kernel_ext.moe, name, None)
    if cls is None:
        print("[5] SKIP: no MXFP4 backend compiled")
    else:
        wp = kt_kernel_ext.WorkerPoolConfig()
        wp.subpool_count = 1
        wp.subpool_numa_map = [0]
        wp.subpool_thread_count = [args.threads]
        cpu_infer = kt_kernel_ext.CPUInfer(wp)
        cfg = kt_kernel_ext.moe.MOEConfig(1, 1, hidden, inter, 0)
        cfg.max_len = 64
        cfg.pool = cpu_infer.backend_
        cfg.quant_config.bits = 4
        cfg.quant_config.group_size = GROUP_SIZE
        cfg.quant_config.zero_point = False
        cfg.gate_projs = [[wdict["gate"][exp].data_ptr()]]
        cfg.up_projs = [[wdict["up"][exp].data_ptr()]]
        cfg.down_projs = [[wdict["down"][exp].data_ptr()]]
        cfg.gate_scales = [[wdict["gate_scale"][exp].data_ptr()]]
        cfg.up_scales = [[wdict["up_scale"][exp].data_ptr()]]
        cfg.down_scales = [[wdict["down_scale"][exp].data_ptr()]]
        moe = cls(cfg)
        p2l = torch.zeros(1, dtype=torch.int64)
        cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
        cpu_infer.sync()

        torch.manual_seed(0)
        for qlen in (1, 33):
            x = (torch.randn((qlen, hidden), dtype=torch.float32) / 10.0).to(torch.bfloat16).contiguous()
            y = torch.empty((qlen, hidden), dtype=torch.bfloat16).contiguous()
            ids = torch.zeros((qlen, 1), dtype=torch.int64).contiguous()
            w = torch.ones((qlen, 1), dtype=torch.float32).contiguous()
            bsz = torch.tensor([qlen], dtype=torch.int32)
            cpu_infer.submit(
                moe.forward_task(bsz.data_ptr(), 1, ids.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), False)
            )
            cpu_infer.sync()
            xf = x.float()
            act = (lambda v: v / (1.0 + torch.exp(-v)))(xf @ deq["gate"].t()) * (xf @ deq["up"].t())
            ref = (act @ deq["down"].t()).to(torch.bfloat16)
            diff = (torch.mean(torch.abs(y.float() - ref.float())) / (torch.mean(torch.abs(ref.float())) + 1e-8)).item()
            status = "OK" if diff < 0.05 else "FAIL"
            print(f"[5] {name} real-expert forward qlen={qlen}: rel-L1={diff:.6f}  [{status}]")
            ok &= diff < 0.05

    print("PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
