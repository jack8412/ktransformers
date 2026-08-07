#!/usr/bin/env python
# coding=utf-8
"""Kimi-K3-shape MoE micro-benchmark: MXFP4 vs FP8 vs AMXINT8.

Runs the same K3 routed-expert shape (hidden/latent 3584, intermediate 3072,
top-16) through AMXFP4_KGroup_MOE, AMXFP8_MOE and AMXInt8_MOE at identical
thread counts, for decode (M=1) and prefill (M=64) regimes, and reports
per-iter latency, tokens/s and effective weight bandwidth (GB/s).

Weight bytes per expert (weights + resident scales):
    MXFP4:  3*I*H * (0.5 + 1/32)   = 0.53125 B/elem (E8M0 group scales in RAM)
    FP8:    3*I*H * (1 + 4/128^2) ~= 1.0   B/elem
    INT8:   3*I*H * 1.0            (per-row scale negligible)

Routing tensors are rotated across iterations so a small hot expert set cannot
sit in LLC at M=1.

Usage:
    python bench/bench_mxfp4_k3.py                       # all available backends
    python bench/bench_mxfp4_k3.py --backends mxfp4,fp8
    python bench/bench_mxfp4_k3.py --experts 256         # trim RAM footprint
"""
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time

import torch

try:
    import kt_kernel_ext
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
    from kt_kernel import kt_kernel_ext  # noqa: E402

# ----- Kimi-K3 routed-expert shape -----
HIDDEN = 3584
INTER = 3072
TOP_K = 16
DEFAULT_EXPERT_NUM = 896
MXFP4_GS = 32
FP8_GS = 128

# M=1: decode. M=64: prefill-sized batch (crosses the mat-mat dispatch boundary
# 4*E/top_k only when experts < 256). M=512: crosses it even at the full E=896.
DEFAULT_M_LIST = [1, 64, 512]
ROUTING_POOL = 32

WORKER_NUMA = 2
WORKER_THREADS_PER_NUMA = 48


def mxfp4_scales_are_resident_bytes():
    """True on builds that keep E8M0 scales resident as 1 B/group. Older builds
    widen them to fp32 at load and want bf16 input, so this also selects which
    scale dtype to synthesize — letting one bench binary A/B both layouts."""
    return hasattr(kt_kernel_ext.moe, "mxfp4_buffer_bytes")


def bytes_per_expert(backend):
    elems = 3 * INTER * HIDDEN
    if backend == "mxfp4":
        # Ask the extension for the true resident size when it can tell us.
        probe = getattr(kt_kernel_ext.moe, "mxfp4_buffer_bytes", None)
        if probe is not None:
            return 2 * probe(INTER, HIDDEN, MXFP4_GS) + probe(HIDDEN, INTER, MXFP4_GS)
        return elems * (0.5 + 4.0 / MXFP4_GS)  # fp32-scale build: 0.625 B/elem
    if backend == "fp8":
        return elems * (1.0 + 4.0 / (FP8_GS * FP8_GS))
    if backend == "int8":
        return elems * 1.0
    raise ValueError(backend)


def synth_mxfp4(expert_num, gen):
    def one(n, k):
        w = torch.randint(0, 256, (expert_num, n, k // 2), generator=gen, dtype=torch.uint8)
        e = torch.randint(118, 124, (expert_num, n, k // MXFP4_GS), generator=gen, dtype=torch.int64)
        if mxfp4_scales_are_resident_bytes():
            s = e.to(torch.uint8).contiguous()  # raw ue8m0 codes
        else:
            s = (e << 7).to(torch.int16).view(torch.bfloat16).contiguous()  # legacy bf16 input
        return w.contiguous(), s

    gw, gs = one(INTER, HIDDEN)
    uw, us = one(INTER, HIDDEN)
    dw, ds = one(HIDDEN, INTER)
    return {"gate_w": gw, "up_w": uw, "down_w": dw, "gate_s": gs, "up_s": us, "down_s": ds}


def synth_fp8(expert_num, gen):
    def one(n, k):
        # Exponent bits 0x30..0x3F keep values sane (no NaN codes matter for perf,
        # but forward output stays finite this way).
        w = (torch.randint(0, 64, (expert_num, n, k), generator=gen, dtype=torch.uint8) | 0x20).contiguous()
        s = (torch.rand((expert_num, n // FP8_GS, k // FP8_GS), generator=gen, dtype=torch.float32) * 0.01).contiguous()
        return w, s

    gw, gs = one(INTER, HIDDEN)
    uw, us = one(INTER, HIDDEN)
    dw, ds = one(HIDDEN, INTER)
    return {"gate_w": gw, "up_w": uw, "down_w": dw, "gate_s": gs, "up_s": us, "down_s": ds}


def synth_bf16(expert_num, gen):
    def one(n, k):
        # Per-expert chunks: avoids a full [E, N, K] fp32 transient (OOM risk on
        # shared hosts) — peak extra memory is one expert's fp32 matrix.
        t = torch.empty((expert_num, n, k), dtype=torch.bfloat16)
        for e in range(expert_num):
            t[e] = (torch.randn((n, k), generator=gen, dtype=torch.float32) / 100).to(torch.bfloat16)
        return t.contiguous()

    return {"gate_w": one(INTER, HIDDEN), "up_w": one(INTER, HIDDEN), "down_w": one(HIDDEN, INTER)}


def build_backend(backend, expert_num, cpu_infer, max_len, gen):
    cfg = kt_kernel_ext.moe.MOEConfig(expert_num, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = max_len
    cfg.pool = cpu_infer.backend_

    if backend == "mxfp4":
        cls = getattr(kt_kernel_ext.moe, "AMXFP4_KGroup_MOE", None)
        weights = synth_mxfp4(expert_num, gen)
        cfg.quant_config.bits = 4
        cfg.quant_config.group_size = MXFP4_GS
        cfg.quant_config.zero_point = False
        cfg.gate_projs = [[t.data_ptr() for t in weights["gate_w"]]]
        cfg.up_projs = [[t.data_ptr() for t in weights["up_w"]]]
        cfg.down_projs = [[t.data_ptr() for t in weights["down_w"]]]
        cfg.gate_scales = [[t.data_ptr() for t in weights["gate_s"]]]
        cfg.up_scales = [[t.data_ptr() for t in weights["up_s"]]]
        cfg.down_scales = [[t.data_ptr() for t in weights["down_s"]]]
    elif backend == "fp8":
        cls = getattr(kt_kernel_ext.moe, "AMXFP8_MOE", None)
        weights = synth_fp8(expert_num, gen)
        cfg.quant_config.bits = 8
        cfg.quant_config.group_size = FP8_GS
        cfg.quant_config.zero_point = False
        cfg.gate_projs = [[t.data_ptr() for t in weights["gate_w"]]]
        cfg.up_projs = [[t.data_ptr() for t in weights["up_w"]]]
        cfg.down_projs = [[t.data_ptr() for t in weights["down_w"]]]
        cfg.gate_scales = [[t.data_ptr() for t in weights["gate_s"]]]
        cfg.up_scales = [[t.data_ptr() for t in weights["up_s"]]]
        cfg.down_scales = [[t.data_ptr() for t in weights["down_s"]]]
    elif backend == "int8":
        cls = getattr(kt_kernel_ext.moe, "AMXInt8_MOE", None)
        weights = synth_bf16(expert_num, gen)
        cfg.gate_proj = weights["gate_w"].data_ptr()
        cfg.up_proj = weights["up_w"].data_ptr()
        cfg.down_proj = weights["down_w"].data_ptr()
        cfg.gate_scale = 0
        cfg.up_scale = 0
        cfg.down_scale = 0
    else:
        raise ValueError(backend)

    if cls is None:
        return None, None
    moe = cls(cfg)
    p2l = torch.arange(expert_num, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
    cpu_infer.sync()
    return moe, weights


def make_routing_pool(M, expert_num, routing, gen):
    pool = []
    for _ in range(ROUTING_POOL):
        if routing == "concentrated":
            hot = torch.randperm(expert_num, generator=gen)[:TOP_K]
            ids = hot.unsqueeze(0).expand(M, TOP_K).contiguous().to(torch.int64)
        else:
            ids = torch.stack([torch.randperm(expert_num, generator=gen)[:TOP_K] for _ in range(M)]).to(torch.int64).contiguous()
        w = torch.rand((M, TOP_K), dtype=torch.float32, generator=gen).contiguous()
        pool.append((ids, w, int(torch.unique(ids).numel())))
    return pool


def bench_one_m(backend, moe, cpu_infer, M, routing, warmup, iters, gen):
    pool = make_routing_pool(M, moe_expert_num, routing, gen)
    x = (torch.randn((M, HIDDEN), generator=gen, dtype=torch.float32) / 100).to(torch.bfloat16).contiguous()
    y = torch.empty((M, HIDDEN), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([M], dtype=torch.int32)

    def run(ids, w):
        cpu_infer.submit(
            moe.forward_task(bsz.data_ptr(), TOP_K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), False)
        )
        cpu_infer.sync()

    for i in range(warmup):
        ids, w, _ = pool[i % ROUTING_POOL]
        run(ids, w)

    unique_sum = 0
    start = time.perf_counter()
    for i in range(iters):
        ids, w, uniq = pool[i % ROUTING_POOL]
        run(ids, w)
        unique_sum += uniq
    total = time.perf_counter() - start

    per_iter_us = total / iters * 1e6
    tok_per_s = M * iters / total
    gbps = unique_sum * bytes_per_expert(backend) / total / 1e9
    # MXFP4/MXFP8 dispatch mat-vec vs mat-mat on qlen > 4 * expert_num / top_k;
    # report which path this M measured so a "prefill" row that stayed on the
    # decode kernel (e.g. M=64 with the full 896 experts -> boundary 224) is
    # visible in the table. FP8 always uses its vec kernel; INT8 always tiles.
    dispatch_boundary = 4 * moe_expert_num / TOP_K
    return {
        "M": M,
        "iters": iters,
        "per_iter_us": per_iter_us,
        "tokens_per_s": tok_per_s,
        "weight_gbps": gbps,
        "avg_unique_experts": unique_sum / iters,
        "mxfp4_path": "mat-mat" if M > dispatch_boundary else "mat-vec",
    }


def get_git_commit():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())
        return {"commit": commit, "dirty": dirty}
    except Exception as e:  # noqa: BLE001
        return {"commit": None, "error": str(e)}


def get_system_info():
    info = {"node": platform.node(), "system": platform.system(), "cpu_cores": os.cpu_count()}
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line:
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
    except Exception:  # noqa: BLE001
        pass
    return info


moe_expert_num = DEFAULT_EXPERT_NUM


def main():
    global moe_expert_num
    p = argparse.ArgumentParser()
    p.add_argument("--backends", type=str, default="mxfp4,fp8,int8")
    p.add_argument("--experts", type=int, default=DEFAULT_EXPERT_NUM)
    p.add_argument("--m-list", type=str, default=",".join(map(str, DEFAULT_M_LIST)))
    p.add_argument("--routing", choices=["balanced", "concentrated"], default="balanced")
    p.add_argument("--numa", type=int, default=WORKER_NUMA)
    p.add_argument("--threads-per-numa", type=int, default=WORKER_THREADS_PER_NUMA)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=300)
    args = p.parse_args()

    moe_expert_num = args.experts
    m_list = [int(x) for x in args.m_list.split(",")]
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]

    print(f"[bench-k3] shape=H{HIDDEN}/I{INTER}/E{args.experts}/top{TOP_K}  routing={args.routing}")
    print(f"[bench-k3] WorkerPool: numa={args.numa} x threads={args.threads_per_numa}  M={m_list}")
    for b in backends:
        print(f"[bench-k3] {b}: {args.experts * bytes_per_expert(b) / 1e9:.1f} GB resident expert weights")

    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = args.numa
    wp.subpool_numa_map = list(range(args.numa))
    wp.subpool_thread_count = [args.threads_per_numa] * args.numa
    cpu_infer = kt_kernel_ext.CPUInfer(wp)

    all_rows = {}
    for backend in backends:
        gen = torch.Generator().manual_seed(0)
        print(f"\n[bench-k3] === {backend}: synthesizing + loading ===")
        t0 = time.time()
        moe, weights = build_backend(backend, args.experts, cpu_infer, max(m_list), gen)
        if moe is None:
            print(f"[bench-k3] {backend}: backend class not available in this build, skipping")
            continue
        print(f"[bench-k3] {backend}: loaded in {time.time()-t0:.1f}s")
        del weights  # C++ copies at load; free the Python-side staging
        gc.collect()

        rows = []
        for M in m_list:
            iters = args.iters if M == 1 else max(50, args.iters // 6)
            r = bench_one_m(backend, moe, cpu_infer, M, args.routing, args.warmup, iters, gen)
            rows.append(r)
            print(
                f"  M={M:>4}  per-iter={r['per_iter_us']:>10.1f} us  tok/s={r['tokens_per_s']:>9.1f}  "
                f"weights={r['weight_gbps']:>7.1f} GB/s  uniq_e={r['avg_unique_experts']:.1f}  "
                f"[mxfp4 dispatch: {r['mxfp4_path']}]"
            )
        all_rows[backend] = rows
        del moe
        gc.collect()

    if len(all_rows) > 1:
        base = next(iter(all_rows))
        print(f"\n=== K3-shape comparison (routing={args.routing}, tok/s ratio vs {base}) ===")
        header = f"{'M':>4}"
        for b in all_rows:
            header += f"  {b + ' us':>12}  {b + ' GB/s':>11}"
        print(header)
        for i, M in enumerate(m_list):
            line = f"{M:>4}"
            for b in all_rows:
                if i < len(all_rows[b]):
                    line += f"  {all_rows[b][i]['per_iter_us']:>12.1f}  {all_rows[b][i]['weight_gbps']:>11.1f}"
                else:
                    line += f"  {'-':>12}  {'-':>11}"
            print(line)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_mxfp4_k3.jsonl")
    record_common = {
        "shape": {"hidden": HIDDEN, "inter": INTER, "expert_num": args.experts, "top_k": TOP_K},
        "worker_pool": {"numa": args.numa, "threads_per_numa": args.threads_per_numa},
        "routing": args.routing,
        "git": get_git_commit(),
        "system": get_system_info(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }
    with open(out_path, "a") as f:
        for b, rows in all_rows.items():
            f.write(json.dumps({"backend": b, "rows": rows, **record_common}) + "\n")
    print(f"\n[bench-k3] appended {len(all_rows)} record(s) -> {out_path}")


if __name__ == "__main__":
    main()
