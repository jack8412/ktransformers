"""Numeric check for REAP norm reporting (TP_MOE_Common::reap_score).

Runs a real MoE forward on CPU and compares the norms kt reports against a
torch reference built from the same weights. What this is guarding:

  - the value is ||f||, the L2 norm of the expert's OUTPUT vector, not of some
    intermediate and not a squared quantity;
  - it is the norm of the vector SUMMED across NUMA partitions. A partition
    holds a slice of the intermediate axis, so its own row is full length but a
    partial value; norming per partition and adding drops the cross terms and
    silently reports something smaller than ||f||;
  - a slot the layer does not own reports 0 rather than stale memory.

Run: python test/test_reap_norms.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build"))

from kt_kernel import kt_kernel_ext  # noqa: E402

EXPERTS = 8
TOP_K = 2
HIDDEN = 1024
INTERMEDIATE = 768  # must divide the NUMA node count; 6 nodes on this host
QLEN = 4
MAX_LEN = 256


def _reference_norms(x, gate, up, down, expert_ids):
    """||down(silu(gate(x)) * up(x))|| per (token, slot), in fp32."""
    out = torch.zeros(QLEN, TOP_K, dtype=torch.float64)
    for i in range(QLEN):
        xi = x[i].float()
        for j in range(TOP_K):
            e = int(expert_ids[i * TOP_K + j])
            g = xi @ gate[e].float().T
            u = xi @ up[e].float().T
            h = torch.nn.functional.silu(g) * u
            f = h @ down[e].float().T
            out[i, j] = f.double().norm()
    return out


def main():
    torch.manual_seed(0)
    cpuinfer = kt_kernel_ext.CPUInfer(64)

    def w(shape):
        return (torch.randn(shape, dtype=torch.float32) / 20.0).to(torch.bfloat16).contiguous()

    gate = w((EXPERTS, INTERMEDIATE, HIDDEN))
    up = w((EXPERTS, INTERMEDIATE, HIDDEN))
    down = w((EXPERTS, HIDDEN, INTERMEDIATE))

    config = kt_kernel_ext.moe.MOEConfig(EXPERTS, TOP_K, HIDDEN, INTERMEDIATE, 0)
    config.max_len = MAX_LEN
    config.gate_proj = gate.data_ptr()
    config.up_proj = up.data_ptr()
    config.down_proj = down.data_ptr()
    config.gate_scale = 0
    config.up_scale = 0
    config.down_scale = 0
    config.pool = cpuinfer.backend_

    moe = kt_kernel_ext.moe.AMXBF16_MOE(config)
    physical_to_logical = torch.arange(EXPERTS, dtype=torch.int64).contiguous()
    cpuinfer.submit(moe.load_weights_task(physical_to_logical.data_ptr()))
    cpuinfer.sync()

    if not hasattr(moe, "set_reap_norms_buffer"):
        raise SystemExit("FAIL: this kt build has no set_reap_norms_buffer")

    norms = torch.zeros(MAX_LEN, TOP_K, dtype=torch.float32).contiguous()
    moe.set_reap_norms_buffer(norms.data_ptr(), MAX_LEN, TOP_K)

    expert_ids = (
        torch.rand(QLEN, EXPERTS).argsort(dim=-1)[:, :TOP_K].reshape(-1).contiguous().to(torch.int64)
    )
    weights = torch.rand(QLEN, TOP_K, dtype=torch.float32).contiguous()
    x = torch.randn(QLEN, HIDDEN, dtype=torch.bfloat16).contiguous()
    y = torch.empty(QLEN, HIDDEN, dtype=torch.bfloat16).contiguous()
    qlen_t = torch.tensor([QLEN], dtype=torch.int32)

    cpuinfer.submit(
        moe.forward_task(
            qlen_t.data_ptr(),
            TOP_K,
            expert_ids.data_ptr(),
            weights.data_ptr(),
            x.data_ptr(),
            y.data_ptr(),
            False,
        )
    )
    cpuinfer.sync()

    got = norms[:QLEN].double()
    want = _reference_norms(x, gate, up, down, expert_ids)

    print(f"{'tok':>3} {'slot':>4} {'expert':>6} {'kt':>12} {'reference':>12} {'rel':>9}")
    worst = 0.0
    for i in range(QLEN):
        for j in range(TOP_K):
            g, r = got[i, j].item(), want[i, j].item()
            rel = abs(g - r) / max(r, 1e-9)
            worst = max(worst, rel)
            print(f"{i:>3} {j:>4} {int(expert_ids[i*TOP_K+j]):>6} {g:>12.5f} {r:>12.5f} {rel:>9.2%}")

    # bf16 weights and an AMX accumulation order that differs from torch's, so
    # this is a numeric-agreement check, not a bitwise one. A norm taken per
    # partition instead of on the summed vector lands far outside this.
    tol = 0.02
    untouched = norms[QLEN:].abs().sum().item()

    print(f"\nworst relative error: {worst:.3%} (tolerance {tol:.1%})")
    print(f"rows past qlen left at zero: {untouched == 0.0}")
    if worst > tol:
        raise SystemExit(f"FAIL: kt norms disagree with the reference by {worst:.3%}")
    if untouched != 0.0:
        raise SystemExit("FAIL: kt wrote past the real row count")
    print("PASS")


if __name__ == "__main__":
    main()
