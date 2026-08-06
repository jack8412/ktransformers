#!/usr/bin/env python
# coding=utf-8
"""MXFP4 MoE correctness at Kimi-K3 expert shapes (manual script, AMX/AVX512 host).

K3 routed experts: hidden (latent) = 3584, intermediate = 3072, top-16 routing,
E2M1 nibbles + ue8m0 group-32 scales along K (compressed-tensors
"mxfp4-pack-quantized"). This script validates, against a fp32 PyTorch
reference built from the exactly-dequantized weights:

  1. expert forward (gate/up/SiLU/down) at K3 shapes across the
     mat-vec / mat-mat dispatch boundary,
  2. GEMM K-edge shapes: K multiples of 32 that are not multiples of 64/128,
  3. a threading / NUMA smoke test at 96 threads over 2 pools.

Note: real Kimi-K3 uses the "situ" activation (beta*tanh(g/beta)*sigmoid(g)*up,
beta=4.0); kt-kernel currently implements SiLU only, so this script tests SiLU.
SiTU support is a planned follow-up.

Usage:
    python examples/test_mxfp4_moe_k3.py                 # correctness sweeps
    python examples/test_mxfp4_moe_k3.py --numa-smoke    # + 96-thread/2-pool smoke
"""
import argparse
import os
import sys
import time

import torch

try:
    import kt_kernel_ext
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__) + "/../build")
    from kt_kernel import kt_kernel_ext

torch.manual_seed(42)

# ----- Kimi-K3 routed-expert shape -----
HIDDEN = 3584  # routed_expert_hidden_size (latent)
INTER = 3072  # moe_intermediate_size
TOP_K = 16
GROUP_SIZE = 32

# Subset of the 896 experts: enough for routing coverage, small enough that the
# fp32 dequantized reference (3 x [E, N, K] fp32) stays a few GB.
EXPERT_NUM = 32

E2M1_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)

THRESHOLD = 0.05  # rel-L1 vs dequantized-weight reference (compute-path noise only)


def pick_backend():
    forced = os.getenv("KT_MXFP4_BACKEND", "").strip().lower()
    if forced == "avx2":
        return "AVX2MXFP4_MOE", kt_kernel_ext.moe.AVX2MXFP4_MOE
    if forced == "amx" or hasattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE"):
        return "AMXFP4_KGroup_MOE", kt_kernel_ext.moe.AMXFP4_KGroup_MOE
    if hasattr(kt_kernel_ext.moe, "AVX2MXFP4_MOE"):
        return "AVX2MXFP4_MOE", kt_kernel_ext.moe.AVX2MXFP4_MOE
    print("SKIP: no MXFP4 backend compiled in this build")
    sys.exit(0)


def make_mxfp4_weight(n, k, gen, exp_lo=-9, exp_hi=-4):
    """Random E2M1 codes + random power-of-two (ue8m0) scales; exact dequant."""
    codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.uint8)
    codes_l = codes.to(torch.int64)
    packed = ((codes_l[:, 1::2] << 4) | codes_l[:, 0::2]).to(torch.uint8).contiguous()
    exps = torch.randint(127 + exp_lo, 127 + exp_hi + 1, (n, k // GROUP_SIZE), generator=gen, dtype=torch.int64)
    scale_bf16 = ((exps << 7).to(torch.int16)).view(torch.bfloat16).contiguous()
    dequant = E2M1_VALUES[codes_l] * scale_bf16.float().repeat_interleave(GROUP_SIZE, dim=1)
    return packed, scale_bf16, dequant


def act_fn(x):
    return x / (1.0 + torch.exp(-x))


def mlp_torch(input_data, gate_proj, up_proj, down_proj):
    gate_buf = torch.mm(input_data, gate_proj.t())
    up_buf = torch.mm(input_data, up_proj.t())
    return torch.mm(act_fn(gate_buf) * up_buf, down_proj.t())


def moe_torch(input_data, expert_ids, weights, gate_proj, up_proj, down_proj, expert_num):
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
        outputs.append(mlp_torch(sorted_tokens[start_idx:end_idx], gate_proj[i], up_proj[i], down_proj[i]))
        start_idx = end_idx
    outs = torch.cat(outputs, dim=0) if outputs else sorted_tokens.new_empty(0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (new_x.view(*expert_ids.shape, -1).float().mul_(weights.unsqueeze(-1)).sum(1)).to(new_x.dtype)


def build_moe(backend_cls, cpu_infer, expert_num, top_k, hidden, inter, weights, max_len):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, top_k, hidden, inter, 0)
    cfg.max_len = max_len
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = GROUP_SIZE
    cfg.quant_config.zero_point = False
    cfg.gate_projs = [[t.data_ptr() for t in weights["gate_w"]]]
    cfg.up_projs = [[t.data_ptr() for t in weights["up_w"]]]
    cfg.down_projs = [[t.data_ptr() for t in weights["down_w"]]]
    cfg.gate_scales = [[t.data_ptr() for t in weights["gate_s"]]]
    cfg.up_scales = [[t.data_ptr() for t in weights["up_s"]]]
    cfg.down_scales = [[t.data_ptr() for t in weights["down_s"]]]
    moe = backend_cls(cfg)
    p2l = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
    cpu_infer.sync()
    return moe


def forward(moe, cpu_infer, qlen, top_k, hidden, expert_ids, weights, input_data):
    output = torch.empty((qlen, hidden), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(
            bsz.data_ptr(), top_k, expert_ids.data_ptr(), weights.data_ptr(), input_data.data_ptr(), output.data_ptr(), False
        )
    )
    cpu_infer.sync()
    return output


def build_weights(expert_num, hidden, inter, gen):
    data = {"gate_w": [], "up_w": [], "down_w": [], "gate_s": [], "up_s": [], "down_s": []}
    deq = {"gate": [], "up": [], "down": []}
    for _ in range(expert_num):
        for proj, (n, k) in (("gate", (inter, hidden)), ("up", (inter, hidden)), ("down", (hidden, inter))):
            p, s, d = make_mxfp4_weight(n, k, gen)
            data[f"{proj}_w"].append(p)
            data[f"{proj}_s"].append(s)
            deq[proj].append(d)
    for proj in ("gate", "up", "down"):
        deq[proj] = torch.stack(deq[proj])
    return data, deq


def rel_l1(output, ref):
    return (torch.mean(torch.abs(output.float() - ref.float())) / (torch.mean(torch.abs(ref.float())) + 1e-8)).item()


def default_cpu_infer(threads):
    nodes = sorted(
        int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()
    )
    count = 2 if len(nodes) >= 2 else 1
    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = count
    wp.subpool_numa_map = nodes[:count]
    wp.subpool_thread_count = [max(1, threads // count)] * count
    return kt_kernel_ext.CPUInfer(wp)


def run_k3_expert_forward(backend_name, backend_cls, threads):
    print(f"\n=== K3 expert forward: E={EXPERT_NUM} top{TOP_K} H{HIDDEN} I{INTER} ({backend_name}, {threads} threads) ===")
    cpu_infer = default_cpu_infer(threads)
    gen = torch.Generator().manual_seed(7)
    t0 = time.time()
    data, deq = build_weights(EXPERT_NUM, HIDDEN, INTER, gen)
    print(f"  weights synthesized in {time.time()-t0:.1f}s")
    moe = build_moe(backend_cls, cpu_infer, EXPERT_NUM, TOP_K, HIDDEN, INTER, data, max_len=4096)

    # qlen 1/4 -> mat-vec; qlen 16/64 -> qlen > 4*E/topk (= 8) mat-mat path.
    failed = False
    for qlen in (1, 4, 16, 64):
        expert_ids = torch.stack([torch.randperm(EXPERT_NUM, generator=gen)[:TOP_K] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, TOP_K), dtype=torch.float32, generator=gen).contiguous()
        x = (torch.randn((qlen, HIDDEN), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
        y = forward(moe, cpu_infer, qlen, TOP_K, HIDDEN, expert_ids, weights, x)
        ref = moe_torch(x.float(), expert_ids, weights, deq["gate"], deq["up"], deq["down"], EXPERT_NUM).to(torch.bfloat16)
        diff = rel_l1(y, ref)
        status = "OK " if diff < THRESHOLD else "FAIL"
        print(f"  qlen={qlen:>3}  rel-L1={diff:.6f}  [{status}]")
        failed |= diff >= THRESHOLD
    return not failed


def run_k_edge_sweep(backend_name, backend_cls, threads):
    """GEMM edges: K multiples of 32 that are not multiples of 64/128, plus the
    real K3 K dims; single expert so any decode/scale indexing slip is visible."""
    print(f"\n=== K-edge sweep ({backend_name}) ===")
    cpu_infer = default_cpu_infer(threads)
    gen = torch.Generator().manual_seed(11)
    failed = False
    for hidden, inter in ((96, 64), (160, 64), (352, 64), (3584, 64), (96, 3072)):
        data, deq = build_weights(1, hidden, inter, gen)
        moe = build_moe(backend_cls, cpu_infer, 1, 1, hidden, inter, data, max_len=64)
        for qlen in (1, 7, 33):  # 7/33: M % 4 != 0 exercises the M-tail path
            expert_ids = torch.zeros((qlen, 1), dtype=torch.int64).contiguous()
            weights = torch.ones((qlen, 1), dtype=torch.float32).contiguous()
            x = (torch.randn((qlen, hidden), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
            y = forward(moe, cpu_infer, qlen, 1, hidden, expert_ids, weights, x)
            ref = mlp_torch(x.float(), deq["gate"][0], deq["up"][0], deq["down"][0]).to(torch.bfloat16)
            diff = rel_l1(y, ref)
            status = "OK " if diff < THRESHOLD else "FAIL"
            print(f"  H={hidden:>4} I={inter:>4} qlen={qlen:>2}  rel-L1={diff:.6f}  [{status}]")
            failed |= diff >= THRESHOLD
    return not failed


def run_numa_smoke(backend_name, backend_cls):
    """96 threads over 2 NUMA pools at K3 shapes: correctness + no deadlock."""
    print(f"\n=== NUMA smoke: 96 threads / 2 pools ({backend_name}) ===")
    nodes = sorted(
        int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()
    )
    if len(nodes) < 2:
        print("  SKIP: host has a single NUMA node")
        return True
    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = 2
    wp.subpool_numa_map = nodes[:2]
    wp.subpool_thread_count = [48, 48]
    cpu_infer = kt_kernel_ext.CPUInfer(wp)

    gen = torch.Generator().manual_seed(13)
    data, deq = build_weights(EXPERT_NUM, HIDDEN, INTER, gen)
    moe = build_moe(backend_cls, cpu_infer, EXPERT_NUM, TOP_K, HIDDEN, INTER, data, max_len=4096)

    failed = False
    for qlen in (1, 64):
        expert_ids = torch.stack([torch.randperm(EXPERT_NUM, generator=gen)[:TOP_K] for _ in range(qlen)]).contiguous()
        weights = torch.rand((qlen, TOP_K), dtype=torch.float32, generator=gen).contiguous()
        x = (torch.randn((qlen, HIDDEN), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
        t0 = time.time()
        iters = 50 if qlen == 1 else 10
        for _ in range(iters):
            y = forward(moe, cpu_infer, qlen, TOP_K, HIDDEN, expert_ids, weights, x)
        dt = (time.time() - t0) / iters * 1e3
        ref = moe_torch(x.float(), expert_ids, weights, deq["gate"], deq["up"], deq["down"], EXPERT_NUM).to(torch.bfloat16)
        diff = rel_l1(y, ref)
        ok = diff < THRESHOLD and torch.isfinite(y.float()).all()
        status = "OK " if ok else "FAIL"
        print(f"  qlen={qlen:>3}  {dt:8.2f} ms/iter  rel-L1={diff:.6f}  [{status}]")
        failed |= not ok
    return not failed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--numa-smoke", action="store_true", help="also run the 96-thread/2-pool smoke test")
    p.add_argument("--skip-edges", action="store_true")
    args = p.parse_args()

    backend_name, backend_cls = pick_backend()
    ok = run_k3_expert_forward(backend_name, backend_cls, args.threads)
    if not args.skip_edges:
        ok &= run_k_edge_sweep(backend_name, backend_cls, args.threads)
    if args.numa_smoke:
        ok &= run_numa_smoke(backend_name, backend_cls)

    print(f"\n{'ALL PASSED' if ok else 'FAILURES DETECTED'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
