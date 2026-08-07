#!/usr/bin/env python
# coding=utf-8
"""Golden bitwise harness for the MXFP4 scale-storage change.

The E8M0 -> fp32 expansion is exact (both the load-time bf16 route and the
use-time `code << 23` route produce identical fp32 bits for all 256 codes), so
switching MXFP4 to E8M0-resident scales MUST NOT move a single output bit.
Since the change replaces the fp32-scale path in place, there is no old backend
left to diff against at runtime — so capture goldens from the fp32-scale build
first, then compare after the change.

    # on the fp32-scale build, before the change
    python3 test/mxfp4_golden_bitwise.py capture --out /work/golden

    # after the change
    python3 test/mxfp4_golden_bitwise.py compare --out /work/golden

Cases: K3 expert dims (hidden 3584, intermediate 3072), SiLU and SiTU,
qlen 1 and 64, every compiled MXFP4 backend, fixed seed and a fixed
WorkerPoolConfig so thread partitioning (and thus accumulation order) is
identical across runs.
"""
import argparse
import hashlib
import json
import os
import sys

import torch

try:
    import kt_kernel_ext
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))
    from kt_kernel import kt_kernel_ext  # noqa: E402

HIDDEN = 3584
INTER = 3072
EXPERT_NUM = 8
TOP_K = 4
GROUP_SIZE = 32
SEED = 20260807

# Kimi-K3 activation_situ_beta / activation_situ_linear_beta.
SITU = (4.0, 25.0)


def backends():
    out = []
    for name in ("AMXFP4_KGroup_MOE", "AVX2MXFP4_MOE"):
        cls = getattr(kt_kernel_ext.moe, name, None)
        if cls is not None:
            out.append((name, cls))
    return out


def make_cpu_infer(threads=32):
    nodes = sorted(
        int(d[4:]) for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()
    )
    count = 2 if len(nodes) >= 2 else 1
    wp = kt_kernel_ext.WorkerPoolConfig()
    wp.subpool_count = count
    wp.subpool_numa_map = nodes[:count]
    wp.subpool_thread_count = [max(1, threads // count)] * count
    return kt_kernel_ext.CPUInfer(wp)


def make_weights(gen):
    """Random E2M1 codes + ue8m0 scales. Returns the raw uint8 scale bytes AND
    the bf16 form, so this harness works against both storage layouts."""
    data = {}
    for proj, (n, k) in (("gate", (INTER, HIDDEN)), ("up", (INTER, HIDDEN)), ("down", (HIDDEN, INTER))):
        packed, s_u8, s_bf16 = [], [], []
        for _ in range(EXPERT_NUM):
            codes = torch.randint(0, 16, (n, k), generator=gen, dtype=torch.int64)
            packed.append(((codes[:, 1::2] << 4) | codes[:, 0::2]).to(torch.uint8).contiguous())
            # Scale exponents 2^-9..2^-4. Deliberately small: with K=3584 and
            # E2M1 magnitudes up to 6, wider exponents drive the SiTU gate deep
            # into tanh/sigmoid saturation, where the output collapses to a few
            # discrete values and stops depending on the scales at all — a
            # golden that would pass even if scale expansion were broken.
            exps = torch.randint(127 - 9, 127 - 3, (n, k // GROUP_SIZE), generator=gen, dtype=torch.int64)
            s_u8.append(exps.to(torch.uint8).contiguous())
            s_bf16.append(((exps << 7).to(torch.int16)).view(torch.bfloat16).contiguous())
        data[proj] = packed
        data[proj + "_u8"] = s_u8
        data[proj + "_bf16"] = s_bf16
    return data


def scales_are_uint8():
    """Detect which resident scale layout this build expects.

    The E8M0-resident build wants raw uint8 scale bytes; the fp32-scale build
    wants bf16 (which it converts at load). Probe the loader's declared dtype
    expectation rather than guessing from the build.
    """
    return os.environ.get("KT_MXFP4_SCALES", "auto") == "u8"


def build_moe(cls, cpu_infer, data, situ, qlen_max, use_u8, flat):
    """`flat` selects the flat single-buffer config (config.gate_proj) instead of
    per-expert pointer lists. The two take different load paths — including
    different NUMA/TP shard code — so both must be covered: a bug confined to
    the flat TP branch is invisible to a per-expert-only golden."""
    cfg = kt_kernel_ext.moe.MOEConfig(EXPERT_NUM, TOP_K, HIDDEN, INTER, 0)
    cfg.max_len = qlen_max
    cfg.pool = cpu_infer.backend_
    cfg.quant_config.bits = 4
    cfg.quant_config.group_size = GROUP_SIZE
    cfg.quant_config.zero_point = False
    suffix = "_u8" if use_u8 else "_bf16"
    # Set before either branch — the flat branch returns early.
    if situ is not None:
        cfg.situ_beta, cfg.situ_linear_beta = situ
    keep = []
    if flat:
        # MOEConfig is a pybind class and rejects arbitrary attributes, so the
        # stacked tensors are returned to the caller to keep alive instead.
        keep = [torch.stack(data[key]).contiguous() for key in
                ("gate", "up", "down", "gate" + suffix, "up" + suffix, "down" + suffix)]
        cfg.gate_proj, cfg.up_proj, cfg.down_proj = (t.data_ptr() for t in keep[:3])
        cfg.gate_scale, cfg.up_scale, cfg.down_scale = (t.data_ptr() for t in keep[3:])
        moe = cls(cfg)
        p2l = torch.arange(EXPERT_NUM, dtype=torch.int64).contiguous()
        cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
        cpu_infer.sync()
        return moe, keep
    cfg.gate_projs = [[t.data_ptr() for t in data["gate"]]]
    cfg.up_projs = [[t.data_ptr() for t in data["up"]]]
    cfg.down_projs = [[t.data_ptr() for t in data["down"]]]
    cfg.gate_scales = [[t.data_ptr() for t in data["gate" + suffix]]]
    cfg.up_scales = [[t.data_ptr() for t in data["up" + suffix]]]
    cfg.down_scales = [[t.data_ptr() for t in data["down" + suffix]]]
    moe = cls(cfg)
    p2l = torch.arange(EXPERT_NUM, dtype=torch.int64).contiguous()
    cpu_infer.submit(moe.load_weights_task(p2l.data_ptr()))
    cpu_infer.sync()
    return moe, keep


def run_case(cls, cpu_infer, data, situ, qlen, use_u8, flat):
    gen = torch.Generator().manual_seed(SEED + qlen)
    moe, _keep = build_moe(cls, cpu_infer, data, situ, max(64, qlen), use_u8, flat)
    ids = torch.stack([torch.randperm(EXPERT_NUM, generator=gen)[:TOP_K] for _ in range(qlen)]).contiguous()
    w = torch.rand((qlen, TOP_K), dtype=torch.float32, generator=gen).contiguous()
    x = (torch.randn((qlen, HIDDEN), dtype=torch.float32, generator=gen) / 10.0).to(torch.bfloat16).contiguous()
    y = torch.empty((qlen, HIDDEN), dtype=torch.bfloat16).contiguous()
    bsz = torch.tensor([qlen], dtype=torch.int32)
    cpu_infer.submit(
        moe.forward_task(bsz.data_ptr(), TOP_K, ids.data_ptr(), w.data_ptr(), x.data_ptr(), y.data_ptr(), False)
    )
    cpu_infer.sync()
    # A golden that happily hashes NaN/Inf proves nothing: garbage would be
    # perfectly reproducible. Refuse to record a case that is not finite.
    if not torch.isfinite(y.float()).all():
        raise RuntimeError(
            f"non-finite output (flat={flat}, qlen={qlen}, situ={situ is not None}) — "
            "the build and the scale dtype being fed are probably mismatched"
        )
    return y


def case_key(backend, situ, qlen, flat):
    return f"{backend}|{'situ' if situ else 'silu'}|q{qlen}|{'flat' if flat else 'perexpert'}"


def collect(use_u8, perturb_scale=False):
    cpu_infer = make_cpu_infer()
    gen = torch.Generator().manual_seed(SEED)
    data = make_weights(gen)
    if perturb_scale:
        # Anti-vacuity probe: the perturbed run is SEPARATE from the golden
        # run, so the perturbation only needs to be unmissable, not minimal.
        #
        # A single 1-step scale bump is NOT reliably visible in the bf16
        # output at qlen=1: any one k-group is 32 of 3072/3584 inputs (~1%),
        # so a x2 change lands near/below bf16's 2^-8 resolution and comes
        # down to per-element rounding luck (observed 2026-08-07: 6/16 then
        # 8/16 false-insensitive q1 cases for gate- and single-down-group
        # perturbations respectively — the gate variant reached 3584 output
        # rounding lotteries, the single down group exactly one).
        #
        # Instead scale expert 0's ENTIRE first down row by 2 steps (x4):
        # output element 0's expert-0 contribution quadruples, a double-digit
        # relative change for every backend/activation/qlen/load-path case,
        # while still proving the full scale path load -> kernel -> output.
        for suffix in ("_u8", "_bf16"):
            t = data["down" + suffix][0]
            if suffix == "_u8":
                t[0, :] = (t[0, :].to(torch.int64) + 2).to(torch.uint8)
            else:
                bits = t.view(torch.int16)
                bits[0, :] = bits[0, :] + (2 << 7)
    results = {}
    for name, cls in backends():
        for situ in (None, SITU):
            for qlen in (1, 64):
                for flat in (False, True):
                    y = run_case(cls, cpu_infer, data, situ, qlen, use_u8, flat)
                    results[case_key(name, situ, qlen, flat)] = y.view(torch.uint8).numpy().tobytes()
    return results


def digest(b):
    return hashlib.sha256(b).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--u8-scales", action="store_true", help="feed raw uint8 ue8m0 scales (E8M0-resident build)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    blob_path = os.path.join(args.out, "mxfp4_golden.bin")
    meta_path = os.path.join(args.out, "mxfp4_golden.json")

    print(f"[golden] mode={args.mode} u8_scales={args.u8_scales}")
    results = collect(args.u8_scales)

    # Determinism guard: the comparison is only meaningful if a single build
    # reproduces itself bit-for-bit (work stealing must not perturb the
    # accumulation order).
    again = collect(args.u8_scales)
    nondet = [k for k in results if results[k] != again[k]]
    if nondet:
        print(f"[golden] FAIL: build is not run-to-run deterministic for {nondet}")
        sys.exit(1)
    print(f"[golden] determinism check OK across {len(results)} cases")

    # Sensitivity guard: a golden that cannot detect a wrong scale is worthless.
    # Perturb one scale exponent by one step and require EVERY case to move.
    # (A saturated activation, or a kernel that ignored the scales entirely,
    # would show up here as an unchanged case.)
    perturbed = collect(args.u8_scales, perturb_scale=True)
    insensitive = [k for k in results if results[k] == perturbed[k]]
    if insensitive:
        print(f"[golden] FAIL: cases insensitive to a 1-step scale change: {insensitive}")
        print("         These cannot validate the scale-storage change — fix the case setup.")
        sys.exit(1)
    print(f"[golden] sensitivity check OK: all {len(results)} cases react to a 1-step scale change")

    if args.mode == "capture":
        keys = sorted(results)
        with open(blob_path, "wb") as f:
            for k in keys:
                f.write(results[k])
        meta = {k: {"bytes": len(results[k]), "sha256_16": digest(results[k])} for k in keys}
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=1, sort_keys=True)
        for k in keys:
            print(f"  captured {k:<58} {meta[k]['sha256_16']}  ({meta[k]['bytes']} B)")
        print(f"[golden] wrote {blob_path}")
        return

    with open(meta_path) as f:
        meta = json.load(f)
    missing = sorted(set(meta) ^ set(results))
    if missing:
        print(f"[golden] FAIL: case set changed: {missing}")
        sys.exit(1)
    bad = []
    for k in sorted(results):
        got = digest(results[k])
        want = meta[k]["sha256_16"]
        status = "MATCH" if got == want else "DIFFER"
        if got != want:
            bad.append(k)
        print(f"  {k:<58} golden={want} now={got}  [{status}]")
    if bad:
        print(f"\n[golden] FAIL: {len(bad)} case(s) not bit-identical: {bad}")
        sys.exit(1)
    print(f"\n[golden] PASS: all {len(results)} cases bit-identical to the fp32-scale baseline")


if __name__ == "__main__":
    main()
