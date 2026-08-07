// Precision test for the Kimi-K3 "situ" activation and its tanh helper.
// Sweeps a wide input range through amx::tanh_avx512 / amx::act_fn and compares
// against a scalar libm reference, reporting worst-case absolute and relative
// error. Also checks that situ_beta == 0 leaves the silu path bit-identical.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../../avx2/avx2_bf16_utils.hpp"
#include "../la/amx.hpp"

namespace {

// Reference: SituAndMul from Kimi-K3 modeling_kimi_linear.py.
float situ_ref(float g, float u, float beta, float linear_beta) {
  float situ_a = beta * std::tanh(g / beta) * (1.0f / (1.0f + std::exp(-g)));
  if (linear_beta > 0.0f) u = linear_beta * std::tanh(u / linear_beta);
  return situ_a * u;
}

float silu_ref(float g, float u) { return g / (1.0f + std::exp(-g)) * u; }

struct ErrStat {
  double max_abs = 0.0;
  double max_rel = 0.0;
  double max_ref = 0.0;
  float worst_in = 0.0f;
  void add(double got, double want, float in) {
    double abs_err = std::fabs(got - want);
    double rel_err = abs_err / std::max(1e-6, std::fabs(want));
    if (abs_err > max_abs) max_abs = abs_err;
    if (std::fabs(want) > max_ref) max_ref = std::fabs(want);
    if (rel_err > max_rel) {
      max_rel = rel_err;
      worst_in = in;
    }
  }
  // Error normalized by the output range of the sweep. Pointwise max_rel is
  // dominated by inputs deep in the sigmoid tail where the output is ~1e-5 of
  // typical magnitude; this is the error a downstream GEMM actually sees.
  double normalized() const { return max_abs / std::max(1e-30, max_ref); }
};

// Inputs spanning the tanh Taylor/exp switch (0.25), saturation, and zero.
std::vector<float> sweep_inputs() {
  std::vector<float> v;
  for (float x = -60.0f; x <= 60.0f; x += 0.013f) v.push_back(x);
  for (float x = -0.5f; x <= 0.5f; x += 0.0007f) v.push_back(x);
  for (float e = -20.0f; e <= 6.0f; e += 0.25f) {
    v.push_back(std::ldexp(1.0f, (int)e));
    v.push_back(-std::ldexp(1.0f, (int)e));
  }
  v.push_back(0.0f);
  for (float sw : {0.125f, 0.25f}) {  // exactly on / astride the branch switch
    v.push_back(sw);
    v.push_back(-sw);
    v.push_back(std::nextafterf(sw, 0.0f));
    v.push_back(std::nextafterf(sw, 1.0f));
  }
  return v;
}

}  // namespace

int main() {
  const std::vector<float> inputs = sweep_inputs();

  // --- tanh ---
  ErrStat tanh_err;
  for (size_t i = 0; i + 16 <= inputs.size(); i += 16) {
    __m512 x = _mm512_loadu_ps(inputs.data() + i);
    alignas(64) float out[16];
    _mm512_storeu_ps(out, amx::tanh_avx512(x));
    for (int j = 0; j < 16; j++) tanh_err.add(out[j], std::tanh(inputs[i + j]), inputs[i + j]);
  }
  printf("tanh_avx512:  max_abs=%.3e  max_rel=%.3e  norm=%.3e (worst rel at x=%g)\n", tanh_err.max_abs,
         tanh_err.max_rel, tanh_err.normalized(), tanh_err.worst_in);

  // --- situ activation, Kimi-K3 params (beta=4, linear_beta=25) ---
  const float beta = 4.0f, linear_beta = 25.0f;
  ErrStat situ_err;
  for (size_t i = 0; i + 16 <= inputs.size(); i += 16) {
    __m512 g = _mm512_loadu_ps(inputs.data() + i);
    // Pair each gate with a shifted up value so both branches see varied input.
    alignas(64) float up_in[16];
    for (int j = 0; j < 16; j++) up_in[j] = inputs[(i + j + 7) % inputs.size()];
    __m512 u = _mm512_loadu_ps(up_in);
    alignas(64) float out[16];
    _mm512_storeu_ps(out, amx::act_fn(g, u, 0.0f, 0.0f, beta, linear_beta));
    for (int j = 0; j < 16; j++) situ_err.add(out[j], situ_ref(inputs[i + j], up_in[j], beta, linear_beta), inputs[i + j]);
  }
  printf("situ act_fn:  max_abs=%.3e  max_rel=%.3e  norm=%.3e (worst rel at gate=%g)\n", situ_err.max_abs,
         situ_err.max_rel, situ_err.normalized(), situ_err.worst_in);

  // --- situ without the linear_beta up-squash ---
  ErrStat situ_nolin_err;
  for (size_t i = 0; i + 16 <= inputs.size(); i += 16) {
    __m512 g = _mm512_loadu_ps(inputs.data() + i);
    __m512 u = _mm512_set1_ps(1.0f);
    alignas(64) float out[16];
    _mm512_storeu_ps(out, amx::act_fn(g, u, 0.0f, 0.0f, beta, 0.0f));
    for (int j = 0; j < 16; j++) situ_nolin_err.add(out[j], situ_ref(inputs[i + j], 1.0f, beta, 0.0f), inputs[i + j]);
  }
  printf("situ (no lin):max_abs=%.3e  max_rel=%.3e  norm=%.3e (worst rel at gate=%g)\n", situ_nolin_err.max_abs,
         situ_nolin_err.max_rel, situ_nolin_err.normalized(), situ_nolin_err.worst_in);

  // --- situ_beta == 0 must be bit-identical to the pre-existing silu path ---
  bool silu_identical = true;
  ErrStat silu_err;
  for (size_t i = 0; i + 16 <= inputs.size(); i += 16) {
    __m512 g = _mm512_loadu_ps(inputs.data() + i);
    __m512 u = _mm512_set1_ps(1.5f);
    alignas(64) float with_situ_param[16], without[16];
    _mm512_storeu_ps(with_situ_param, amx::act_fn(g, u, 0.0f, 0.0f, 0.0f, 0.0f));
    _mm512_storeu_ps(without, amx::act_fn(g, u, 0.0f, 0.0f));
    for (int j = 0; j < 16; j++) {
      if (with_situ_param[j] != without[j]) silu_identical = false;
      silu_err.add(with_situ_param[j], silu_ref(inputs[i + j], 1.5f), inputs[i + j]);
    }
  }
  printf("silu passthrough: %s (vs libm max_abs=%.3e)\n", silu_identical ? "bit-identical" : "DIVERGED",
         silu_err.max_abs);

  // --- AVX2 vector path vs its own scalar tail ---
  // avx2/moe_base.hpp processes 8 lanes with avx2::act_fn and finishes the row
  // with a libm scalar loop; the two must agree or one row would be computed
  // two different ways. Python-level tests rarely reach the tail because the
  // MoE dims are 32-aligned, so check it directly here.
  ErrStat avx2_vec_err, avx2_tail_err;
  for (size_t i = 0; i + 8 <= inputs.size(); i += 8) {
    alignas(32) float up_in[8];
    for (int j = 0; j < 8; j++) up_in[j] = inputs[(i + j + 7) % inputs.size()];
    __m256 g = _mm256_loadu_ps(inputs.data() + i);
    __m256 u = _mm256_loadu_ps(up_in);
    alignas(32) float out[8];
    _mm256_storeu_ps(out, avx2::act_fn(g, u, 0.0f, 0.0f, beta, linear_beta));
    for (int j = 0; j < 8; j++) {
      double want = situ_ref(inputs[i + j], up_in[j], beta, linear_beta);
      avx2_vec_err.add(out[j], want, inputs[i + j]);
      // Exactly the expression in the avx2/moe_base.hpp scalar tail.
      float gs = inputs[i + j], us = up_in[j];
      float situ_a = beta * tanhf(gs / beta) / (1.0f + expf(-gs));
      float us2 = linear_beta * tanhf(us / linear_beta);
      avx2_tail_err.add(out[j], (double)(situ_a * us2), inputs[i + j]);
    }
  }
  printf("avx2 situ:    max_rel=%.3e  norm=%.3e (vs libm ref)\n", avx2_vec_err.max_rel, avx2_vec_err.normalized());
  printf("avx2 vec-vs-scalar-tail: max_rel=%.3e  norm=%.3e\n", avx2_tail_err.max_rel, avx2_tail_err.normalized());

  // `norm` (error / output range) is the primary bound — with linear_beta=25 the
  // situ output reaches ~100, so absolute error scales with magnitude, while
  // pointwise max_rel is dominated by the sigmoid tail where the output is ~1e-5
  // of typical magnitude and its absolute error is ~1e-10. The kernel rounds this
  // result straight to bf16 (~2e-3 relative), so 1e-6 is far below what the
  // following store can represent. The floor is the shared exp_avx512
  // polynomial, which the pre-existing silu path also rides on.
  // Measured on Granite Rapids: tanh norm 8.6e-7, situ norm 1.2e-6. The bounds
  // below leave ~4x headroom over that floor and are still ~400x tighter than
  // bf16, so a wrong constant, a wrong branch, or a dropped linear_beta (which
  // move the error by orders of magnitude) all fail loudly.
  bool ok = tanh_err.normalized() < 2e-6 && tanh_err.max_rel < 5e-6 && situ_err.normalized() < 5e-6 &&
            situ_err.max_rel < 2e-5 && situ_nolin_err.normalized() < 5e-6 && silu_identical &&
            avx2_vec_err.normalized() < 5e-6 && avx2_tail_err.normalized() < 5e-6;
  printf("%s\n", ok ? "PASS: situ activation matches the scalar reference" : "FAIL");
  return ok ? 0 : 1;
}
