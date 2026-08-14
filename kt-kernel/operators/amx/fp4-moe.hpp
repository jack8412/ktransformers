/**
 * @Description  : MXFP4 MoE operator — FP4 E2M1 weights × BF16 activations
 * @Author       : oql, Codex and Claude
 * @Date         : 2026-04-20
 * @Version      : 1.0.0
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 *
 * Based on k2-moe.hpp (RAWINT4). Key differences from RAWINT4:
 *   Weight:   FP4 E2M1 (nibble-packed, same layout) → PSHUFB lookup → BF16
 *   Act:      BF16 direct (BufferABF16Impl, no online INT8 quantization)
 *   Dot prod: _mm512_dpbf16_ps (BF16×BF16→FP32) instead of _mm512_dpbssd_epi32
 *   Scale:    E8M0 per-group byte, resident as-is, expanded to FP32 at use
 **/
#ifndef CPUINFER_OPERATOR_AMX_FP4_MOE_H
#define CPUINFER_OPERATOR_AMX_FP4_MOE_H

#include <atomic>
#include <cstdlib>
#include <map>
#include <mutex>
#include <utility>
#include "la/amx_raw_buffers.hpp"  // BufferABF16Impl
#include "moe_base.hpp"

// Staging arenas for weight loading, reused across layers.
//
// Every MoE layer of a given model has identical expert geometry, so the
// previous per-layer `new uint8_t[]` paid first-touch page faults on several
// GB of fresh anonymous memory once per layer -- 92 times over on Kimi-K3,
// for buffers of the same shape that were freed moments later. Reuse keeps
// the pages hot and removes that cost entirely.
//
// NUMA placement is preserved by construction: the first allocation happens
// on the owning socket's thread inside do_numa_job, so first touch binds the
// pages to that socket, and every later layer reuses the same arena from the
// same socket. The mutex serialises only the (rare) allocation, never the
// copies.
inline uint8_t* kt_staging_arena(int numa_id, int which, size_t bytes) {
  static std::mutex mu;
  static std::map<std::pair<int, int>, std::pair<uint8_t*, size_t>> arenas;
  std::lock_guard<std::mutex> lock(mu);
  auto key = std::make_pair(numa_id, which);
  auto it = arenas.find(key);
  if (it != arenas.end()) {
    if (it->second.second >= bytes) return it->second.first;
    delete[] it->second.first;
    arenas.erase(it);
  }
  uint8_t* p = new uint8_t[bytes];
  arenas.emplace(key, std::make_pair(p, bytes));
  return p;
}


namespace amx {

// ============================================================================
// Resident MXFP4 weight buffer: nibble-packed E2M1 + raw E8M0 scale bytes.
//
// Deliberately NOT BufferBInt4KGroupImpl (which RAWINT4 uses): RAWINT4's group
// scales are arbitrary bf16 values that cannot be represented in one byte,
// while MXFP4's are power-of-two E8M0 codes. Keeping them as 1 B/group and
// expanding to fp32 in the kernel epilogue costs 0.53125 B/elem resident
// instead of 0.625 (-15%), and the expansion is exact so outputs are unchanged.
// ============================================================================
template <typename K>
struct BufferBMXFP4KGroupImpl {
  using dt = typename K::dt;
  dt* b;       // nibble-packed E2M1 weights, row-major [n, k/2]
  uint8_t* d;  // E8M0 exponent codes, row-major [n, k/k_group_size]
  int n, k, k_group_size, k_group_count;

  static constexpr int N_STEP = K::N_STEP;
  static constexpr int K_STEP = K::K_STEP;
  static constexpr bool SCALE = true;

  // Scales start on a 64B boundary. n%N_STEP==0 and k%K_STEP==0 already make
  // n*k/2 a multiple of 512, but round explicitly so the invariant is local.
  static size_t scale_offset(int n, int k) {
    return ((static_cast<size_t>(n) * static_cast<size_t>(k) / 2) + 63) & ~static_cast<size_t>(63);
  }

  static size_t required_size(int n, int k, int k_group_size) {
    // Round the total to 64 as well: the caller hands this straight to
    // std::aligned_alloc, whose size must be a multiple of the alignment.
    // (The scale region is n*(k/gs) bytes, which need not be 64-aligned.)
    size_t total = scale_offset(n, k) + static_cast<size_t>(n) * (k / k_group_size);
    return (total + 63) & ~static_cast<size_t>(63);
  }

  BufferBMXFP4KGroupImpl(int n, int k, int k_group_size, void* ptr) : n(n), k(k), k_group_size(k_group_size) {
    assert(reinterpret_cast<intptr_t>(ptr) % 64 == 0);
    if (n % N_STEP || k % K_STEP || k % k_group_size) {
      printf("BufferBMXFP4KGroupImpl: n=%d k=%d N_STEP=%d K_STEP=%d gs=%d\n", n, k, N_STEP, K_STEP, k_group_size);
      throw std::runtime_error("n or k is not aligned to N_STEP or K_STEP");
    }
    k_group_count = k / k_group_size;
    b = reinterpret_cast<dt*>(ptr);
    d = reinterpret_cast<uint8_t*>(ptr) + scale_offset(n, k);
  }

  void from_raw_mat(const uint8_t* proj, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    const size_t rows = static_cast<size_t>(n_end - n_start);
    std::memcpy(reinterpret_cast<uint8_t*>(b) + n_start * row_bytes, proj + n_start * row_bytes, rows * row_bytes);
  }

  // Raw E8M0 scale bytes: copied verbatim, expanded at use time.
  void scales_from_raw(const uint8_t* src, int ith, int nth) {
    auto [n_start, n_end] = K::split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const size_t row = static_cast<size_t>(k_group_count);
    std::memcpy(d + n_start * row, src + n_start * row, static_cast<size_t>(n_end - n_start) * row);
  }

  dt* get_submat(int n, int k, int n_begin, int k_begin) {
    const size_t row_bytes = static_cast<size_t>(k) / 2;
    return reinterpret_cast<dt*>(reinterpret_cast<uint8_t*>(b) + static_cast<size_t>(n_begin) * row_bytes +
                                 static_cast<size_t>(k_begin) / 2);
  }

  const uint8_t* get_scale(int /*n*/, int n_begin, int k, int k_begin) const {
    return d + static_cast<size_t>(n_begin) * (k / k_group_size) + k_begin / k_group_size;
  }

  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_per_thread = (n + nth - 1) / nth;
    n_per_thread = (n_per_thread + N_STEP - 1) / N_STEP * N_STEP;
    int n_start = std::min(ith * n_per_thread, n);
    int n_end = std::min(n_start + n_per_thread, n);
    return {n_start, n_end};
  }
};

// ============================================================================
// MXFP4 kernel: FP4 E2M1 weights × BF16 activations → FP32 output (AVX512)
// ============================================================================
struct GemmKernel224MXFP4SmallKGroup {
  using dt = uint8_t;
  using output_t = float;
  static constexpr double ELEMENT_SIZE = 0.5;

  static const int M_STEP = 1;
  static const int N_STEP = 32;
  static const int K_STEP = 32;

  static inline const int N_BLOCK = 256;
  static inline const int K_BLOCK = 7168;

  static std::string name() { return "MXFP4_KGROUP"; }
  static int recommended_nth(int n) { return (n + N_BLOCK - 1) / N_BLOCK; }
  static std::pair<int, int> split_range_n(int n, int ith, int nth) {
    int n_start = N_BLOCK * ith;
    int n_end = std::min(n, N_BLOCK * (ith + 1));
    return {n_start, n_end};
  }
  static void config() {}

  // FP4 E2M1 → BF16 LUTs (16 entries each, for PSHUFB within 128-bit lanes)
  // E2M1 values: {0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0}
  alignas(16) static constexpr uint8_t fp4_bf16_lo[16] = {
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0,   //  0..7  positive
      0x00, 0x00, 0x80, 0xC0, 0x00, 0x40, 0x80, 0xC0};  //  8..15 negative
  alignas(16) static constexpr uint8_t fp4_bf16_hi[16] = {
      0x00, 0x3F, 0x3F, 0x3F, 0x40, 0x40, 0x40, 0x40,   //  0..7  positive
      0x80, 0xBF, 0xBF, 0xBF, 0xC0, 0xC0, 0xC0, 0xC0};  //  8..15 negative

  // Convert 16 packed FP4 bytes (32 values = 1 k_group) → 32 BF16 values (__m512i)
  // Output column order: [BF16(lo[0]),BF16(hi[0]), ..., BF16(lo[15]),BF16(hi[15])]
  __attribute__((always_inline)) static inline __m512i mxfp4_to_bf16_32(__m128i packed) {
    __m128i lo_mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(packed, lo_mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), lo_mask);

    __m128i lut_lo = _mm_load_si128((__m128i*)fp4_bf16_lo);
    __m128i lut_hi = _mm_load_si128((__m128i*)fp4_bf16_hi);

    // Look up low/high bytes for lo nibbles → 16 BF16 values
    __m128i l_lo = _mm_shuffle_epi8(lut_lo, lo);
    __m128i l_hi = _mm_shuffle_epi8(lut_hi, lo);
    __m128i lo_bf16_0 = _mm_unpacklo_epi8(l_lo, l_hi);  // BF16(lo[0..7])
    __m128i lo_bf16_1 = _mm_unpackhi_epi8(l_lo, l_hi);  // BF16(lo[8..15])

    // Look up low/high bytes for hi nibbles → 16 BF16 values
    __m128i h_lo = _mm_shuffle_epi8(lut_lo, hi);
    __m128i h_hi = _mm_shuffle_epi8(lut_hi, hi);
    __m128i hi_bf16_0 = _mm_unpacklo_epi8(h_lo, h_hi);  // BF16(hi[0..7])
    __m128i hi_bf16_1 = _mm_unpackhi_epi8(h_lo, h_hi);  // BF16(hi[8..15])

    // Interleave lo/hi at 16-bit: [lo[0],hi[0], lo[1],hi[1], ...] = column order
    __m128i p0 = _mm_unpacklo_epi16(lo_bf16_0, hi_bf16_0);  // cols  0..7
    __m128i p1 = _mm_unpackhi_epi16(lo_bf16_0, hi_bf16_0);  // cols  8..15
    __m128i p2 = _mm_unpacklo_epi16(lo_bf16_1, hi_bf16_1);  // cols 16..23
    __m128i p3 = _mm_unpackhi_epi16(lo_bf16_1, hi_bf16_1);  // cols 24..31

    __m256i q0 = _mm256_inserti128_si256(_mm256_castsi128_si256(p0), p1, 1);
    __m256i q1 = _mm256_inserti128_si256(_mm256_castsi128_si256(p2), p3, 1);
    return _mm512_inserti64x4(_mm512_castsi256_si512(q0), q1, 1);
  }

  struct ActivationBF16 {
    __m512bh a;
#if !defined(__AVX512BF16__)
    __m512 a_even;
    __m512 a_odd;
    inline static const __m512i odd_mask = _mm512_set1_epi32(0xFFFF0000);
#endif

    __attribute__((always_inline)) ActivationBF16(__m512bh a_) : a(a_) {
#if !defined(__AVX512BF16__)
      a_even = _mm512_castsi512_ps(_mm512_slli_epi32((__m512i)a_, 16));
      a_odd = _mm512_castsi512_ps(_mm512_and_si512((__m512i)a_, odd_mask));
#endif
    }
  };

  struct DequantizedWeight {
#if defined(__AVX512BF16__)
    __m512bh d;
#else
    __m512 w_even;
    __m512 w_odd;
    inline static const __m128i lo_mask = _mm_set1_epi8(0x0F);
    inline static const __m512 lut = _mm512_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f, -0.0f, -0.5f, -1.0f,
                                                    -1.5f, -2.0f, -3.0f, -4.0f, -6.0f);
#endif

    __attribute__((always_inline)) DequantizedWeight(__m128i w) {
#if defined(__AVX512BF16__)
      d = (__m512bh)mxfp4_to_bf16_32(w);
#else
      __m128i lo = _mm_and_si128(w, lo_mask);
      __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), lo_mask);

      __m512i lo_32 = _mm512_cvtepu8_epi32(lo);
      __m512i hi_32 = _mm512_cvtepu8_epi32(hi);

      w_even = _mm512_permutexvar_ps(lo_32, lut);
      w_odd = _mm512_permutexvar_ps(hi_32, lut);
#endif
    }
  };

  __attribute__((always_inline)) static inline __m512 mxfp4_dot_bf16(const DequantizedWeight& w,
                                                                     const ActivationBF16& act) {
#if defined(__AVX512BF16__)
    return _mm512_dpbf16_ps(_mm512_setzero_ps(), act.a, w.d);
#else
    __m512 dot = _mm512_mul_ps(act.a_odd, w.w_odd);
    return _mm512_fmadd_ps(act.a_even, w.w_even, dot);
#endif
  }

  // Buffers
  using BufferA = BufferABF16Impl<GemmKernel224MXFP4SmallKGroup>;         // raw BF16, no quant
  using BufferB = BufferBMXFP4KGroupImpl<GemmKernel224MXFP4SmallKGroup>;  // FP4 nibbles + E8M0 scales
  using BufferC = BufferCReduceImpl<GemmKernel224MXFP4SmallKGroup>;      // FP32 reduce

  // 4 个 zmm 的 horizontal reduce → 4 个连续 fp32。
  // 4 次 reduce_add_ps 之间无依赖，编译器/CPU 可并行调度。
  __attribute__((always_inline)) static inline void reduce4(__m512 s0, __m512 s1, __m512 s2, __m512 s3, float* dst) {
    dst[0] = _mm512_reduce_add_ps(s0);
    dst[1] = _mm512_reduce_add_ps(s1);
    dst[2] = _mm512_reduce_add_ps(s2);
    dst[3] = _mm512_reduce_add_ps(s3);
  }

  // mat-vec: M 个独立 token，N 维 4 行一组累加，摊销 horizontal reduce。
  static void fp4_mat_vec_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;

    for (int m_idx = 0; m_idx < m; m_idx++) {
      float* c_row = bc->get_submat(m, n, m_idx, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, m_idx, 0);

      int n_pos = n_start;
      // 主循环: N 维 4 行一组
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        __m512 acc2 = _mm512_setzero_ps();
        __m512 acc3 = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          acc0 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s0[g])), mxfp4_dot_bf16(d0, a), acc0);
          acc1 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s1[g])), mxfp4_dot_bf16(d1, a), acc1);
          acc2 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s2[g])), mxfp4_dot_bf16(d2, a), acc2);
          acc3 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s3[g])), mxfp4_dot_bf16(d3, a), acc3);
        }
        reduce4(acc0, acc1, acc2, acc3, c_row + (n_pos - n_start));
      }
      // N 尾巴: N % 4 != 0 时单行 fallback
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s[g])), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }

  // mat-mat: 4×4 register tile (M_TILE=4, N_TILE=4 → 16 累加器)。
  // 每 K-group 解码 4 行 N 一次, 被 4 个 token 共享 → PSHUFB 解码开销 / 4。
  // M / N 尾巴回退到 mat-vec 单 token 内层 (V4 chunked-prefill 16/32/64 整数倍, 极少触发)。
  static void fp4_mat_mat_kgroup(int m, int n, int k, int k_group_size, BufferA* ba, BufferB* bb, BufferC* bc, int ith,
                                 int nth) {
    auto [n_start, n_end] = split_range_n(n, ith, nth);
    if (n_start >= n_end) return;
    const int kg_count = k / 32;
    constexpr int MB = 4;
    constexpr int NB = 4;

    int m_pos = 0;
    for (; m_pos + MB <= m; m_pos += MB) {
      __m512bh* a_rows[MB] = {
          (__m512bh*)ba->get_submat(m, k, m_pos + 0, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 1, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 2, 0),
          (__m512bh*)ba->get_submat(m, k, m_pos + 3, 0),
      };

      int n_pos = n_start;
      for (; n_pos + NB <= n_end; n_pos += NB) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);

        __m512 acc[MB][NB];
        for (int i = 0; i < MB; i++)
          for (int j = 0; j < NB; j++) acc[i][j] = _mm512_setzero_ps();

        for (int g = 0; g < kg_count; g++) {
          // 4 行权重解码一次, MB 个 token 共享
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          const __m512 sv0 = _mm512_set1_ps(e8m0_to_fp32(s0[g]));
          const __m512 sv1 = _mm512_set1_ps(e8m0_to_fp32(s1[g]));
          const __m512 sv2 = _mm512_set1_ps(e8m0_to_fp32(s2[g]));
          const __m512 sv3 = _mm512_set1_ps(e8m0_to_fp32(s3[g]));

#define V_FMA_ROW(M_I)                                                      \
  do {                                                                      \
    const ActivationBF16 a(a_rows[M_I][g]);                                 \
    acc[M_I][0] = _mm512_fmadd_ps(sv0, mxfp4_dot_bf16(d0, a), acc[M_I][0]); \
    acc[M_I][1] = _mm512_fmadd_ps(sv1, mxfp4_dot_bf16(d1, a), acc[M_I][1]); \
    acc[M_I][2] = _mm512_fmadd_ps(sv2, mxfp4_dot_bf16(d2, a), acc[M_I][2]); \
    acc[M_I][3] = _mm512_fmadd_ps(sv3, mxfp4_dot_bf16(d3, a), acc[M_I][3]); \
  } while (0)
          V_FMA_ROW(0);
          V_FMA_ROW(1);
          V_FMA_ROW(2);
          V_FMA_ROW(3);
#undef V_FMA_ROW
        }
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          reduce4(acc[i][0], acc[i][1], acc[i][2], acc[i][3], c_row + (n_pos - n_start));
        }
      }
      // N 尾巴: 单 N 列 × MB token (V4 不触发)
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        for (int i = 0; i < MB; i++) {
          float* c_row = bc->get_submat(m, n, m_pos + i, n_start);
          __m512 acc = _mm512_setzero_ps();
          for (int g = 0; g < kg_count; g++) {
            const ActivationBF16 a(a_rows[i][g]);
            const DequantizedWeight d(w[g]);
            acc = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s[g])), mxfp4_dot_bf16(d, a), acc);
          }
          c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
        }
      }
    }
    // M 尾巴: M 不是 MB 倍数时余下 token, 退回单 token mat-vec 内层 (V4 不触发)
    for (int mi = m_pos; mi < m; mi++) {
      float* c_row = bc->get_submat(m, n, mi, n_start);
      __m512bh* a_row = (__m512bh*)ba->get_submat(m, k, mi, 0);
      int n_pos = n_start;
      for (; n_pos + 4 <= n_end; n_pos += 4) {
        __m128i* w0 = (__m128i*)bb->get_submat(n, k, n_pos + 0, 0);
        __m128i* w1 = (__m128i*)bb->get_submat(n, k, n_pos + 1, 0);
        __m128i* w2 = (__m128i*)bb->get_submat(n, k, n_pos + 2, 0);
        __m128i* w3 = (__m128i*)bb->get_submat(n, k, n_pos + 3, 0);
        const uint8_t* s0 = bb->get_scale(n, n_pos + 0, k, 0);
        const uint8_t* s1 = bb->get_scale(n, n_pos + 1, k, 0);
        const uint8_t* s2 = bb->get_scale(n, n_pos + 2, k, 0);
        const uint8_t* s3 = bb->get_scale(n, n_pos + 3, k, 0);
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps(), a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d0(w0[g]);
          const DequantizedWeight d1(w1[g]);
          const DequantizedWeight d2(w2[g]);
          const DequantizedWeight d3(w3[g]);
          a0 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s0[g])), mxfp4_dot_bf16(d0, a), a0);
          a1 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s1[g])), mxfp4_dot_bf16(d1, a), a1);
          a2 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s2[g])), mxfp4_dot_bf16(d2, a), a2);
          a3 = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s3[g])), mxfp4_dot_bf16(d3, a), a3);
        }
        reduce4(a0, a1, a2, a3, c_row + (n_pos - n_start));
      }
      for (; n_pos < n_end; n_pos++) {
        __m128i* w = (__m128i*)bb->get_submat(n, k, n_pos, 0);
        const uint8_t* s = bb->get_scale(n, n_pos, k, 0);
        __m512 acc = _mm512_setzero_ps();
        for (int g = 0; g < kg_count; g++) {
          const ActivationBF16 a(a_row[g]);
          const DequantizedWeight d(w[g]);
          acc = _mm512_fmadd_ps(_mm512_set1_ps(e8m0_to_fp32(s[g])), mxfp4_dot_bf16(d, a), acc);
        }
        c_row[n_pos - n_start] = _mm512_reduce_add_ps(acc);
      }
    }
  }
};

// Dispatch functions
inline void vec_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroup::fp4_mat_vec_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

inline void mat_mul_kgroup(int m, int n, int k, int k_group_size,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferA> ba,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferB> bb,
                           std::shared_ptr<GemmKernel224MXFP4SmallKGroup::BufferC> bc, int ith, int nth) {
  GemmKernel224MXFP4SmallKGroup::fp4_mat_mat_kgroup(m, n, k, k_group_size, ba.get(), bb.get(), bc.get(), ith, nth);
}

}  // namespace amx

// ============================================================================
// AMX_FP4_MOE_TP — CRTP class, identical structure to AMX_K2_MOE_TP
// ============================================================================
template <class T = amx::GemmKernel224MXFP4SmallKGroup>
class AMX_FP4_MOE_TP : public AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>> {
  using Base = AMX_MOE_BASE<T, AMX_FP4_MOE_TP<T>>;
  using Base::config_;
  using Base::down_ba_;
  using Base::down_bb_;
  using Base::down_bc_;
  using Base::gate_bb_;
  using Base::gate_bc_;
  using Base::gate_up_ba_;
  using Base::m_local_num_;
  using Base::tp_part_idx;
  using Base::up_bb_;
  using Base::up_bc_;

 public:
  using typename Base::input_t;
  using typename Base::output_t;

  AMX_FP4_MOE_TP() = default;
  AMX_FP4_MOE_TP(GeneralMOEConfig config, int tp_part_idx_ = 0) : Base(config, tp_part_idx_) {}

  void derived_init() {
    auto& quant_config = config_.quant_config;
    // fp4_mat_vec_kgroup / fp4_mat_mat_kgroup hard-code kg_count = k / 32, so any
    // other group_size would silently read the wrong scales.
    if (quant_config.group_size != 32 || quant_config.zero_point) {
      throw std::runtime_error("MXFP4 MoE requires group_size == 32 and no zero_point");
    }
    if (config_.hidden_size % quant_config.group_size != 0 ||
        config_.intermediate_size % quant_config.group_size != 0) {
      throw std::runtime_error("MXFP4 MoE: hidden_size and intermediate_size must be divisible by group_size");
    }
    printf("Creating AMX_FP4_MOE_TP %d at numa %d\n", tp_part_idx, numa_node_of_cpu(sched_getcpu()));
  }

  ~AMX_FP4_MOE_TP() = default;

  // BufferA: raw BF16, no group_size needed
  size_t buffer_a_required_size_impl(size_t m, size_t k) const { return T::BufferA::required_size(m, k); }
  size_t buffer_b_required_size_impl(size_t n, size_t k) const {
    return T::BufferB::required_size(n, k, config_.quant_config.group_size);
  }
  size_t buffer_c_required_size_impl(size_t m, size_t n) const { return T::BufferC::required_size(m, n); }

  std::shared_ptr<typename T::BufferA> make_buffer_a_impl(size_t m, size_t k, void* data) const {
    return std::make_shared<typename T::BufferA>(m, k, data);
  }
  std::shared_ptr<typename T::BufferB> make_buffer_b_impl(size_t n, size_t k, void* data) const {
    return std::make_shared<typename T::BufferB>(n, k, config_.quant_config.group_size, data);
  }
  std::shared_ptr<typename T::BufferC> make_buffer_c_impl(size_t m, size_t n, void* data) const {
    return std::make_shared<typename T::BufferC>(m, n, data);
  }

  void do_gate_up_gemm(bool do_up, int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];
    auto& ba = gate_up_ba_[expert_idx];
    auto& bb = do_up ? up_bb_[expert_idx] : gate_bb_[expert_idx];
    auto& bc = do_up ? up_bc_[expert_idx] : gate_bc_[expert_idx];

    if (use_mat_mul(qlen)) {
      amx::mat_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.intermediate_size, config_.hidden_size, group_size, ba, bb, bc, ith, nth);
    }
  }

  // The default heuristic (qlen > 4*E/topk = 224 for K3) tunes for plain
  // prefill, where per-expert row counts track batch qlen.  Speculative
  // VERIFY batches sit far below it (bs x draft_tokens <= ~232) yet re-read
  // every expert's weights per token on the vec path — the mat path
  // amortizes the weight reads exactly there.  KT_MOE_MATMUL_MIN_QLEN
  // overrides the switch point; unset keeps the historical behavior
  // (and the golden gate's bitwise pin) untouched.
  bool use_mat_mul(int qlen) const {
    static const int min_qlen = []() {
      const char* env = std::getenv("KT_MOE_MATMUL_MIN_QLEN");
      return env ? std::atoi(env) : -1;
    }();
    if (min_qlen >= 0) return qlen >= min_qlen;
    return qlen > 4 * config_.expert_num / config_.num_experts_per_tok;
  }

  void do_down_gemm(int expert_idx, int ith, int nth, int qlen) {
    auto& group_size = config_.quant_config.group_size;
    int m = m_local_num_[expert_idx];

    if (use_mat_mul(qlen)) {
      amx::mat_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    } else {
      amx::vec_mul_kgroup(m, config_.hidden_size, config_.intermediate_size, group_size, down_ba_[expert_idx],
                          down_bb_[expert_idx], down_bc_[expert_idx], ith, nth);
    }
  }

  /// \brief Hand one expert's weight buffers to another and refill them.
  ///
  /// Cold-only residency holds a BufferB only for experts this CPU serves, so
  /// a swap that DEMOTES a GPU expert to the CPU finds no buffer there. Swaps
  /// are 1:1, so the promoted expert's storage is exactly what the demoted one
  /// needs: this moves the three shared_ptrs rather than allocating, which
  /// keeps the CPU-held count -- and therefore RSS -- invariant for the life
  /// of the process.
  ///
  /// `gate`/`up`/`down` and their scales are the FULL (unsliced) expert as it
  /// sits in the checkpoint. The NUMA slicing below mirrors the bulk load
  /// exactly: gate and up are row slices, down is column-strided.
  void install_expert_from_raw(int promote_id, int demote_id, intptr_t gate, intptr_t up, intptr_t down,
                               intptr_t gate_scale, intptr_t up_scale, intptr_t down_scale, int full_intermediate) {
    const int group_size = config_.quant_config.group_size;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;  // this partition's share
    const size_t wec = I * H;                    // elements, nibble-packed
    const size_t sec = (H / group_size) * I;     // scale bytes for gate/up
    const size_t i = (size_t)tp_part_idx;

    if (gate_bb_[promote_id] == nullptr)
      throw std::runtime_error("install_expert_from_raw: promoted expert holds no CPU buffer");
    if (gate_bb_[demote_id] != nullptr)
      throw std::runtime_error("install_expert_from_raw: demoted expert already holds a CPU buffer");

    gate_bb_[demote_id] = std::move(gate_bb_[promote_id]);
    up_bb_[demote_id] = std::move(up_bb_[promote_id]);
    down_bb_[demote_id] = std::move(down_bb_[promote_id]);
    gate_bb_[promote_id] = nullptr;
    up_bb_[promote_id] = nullptr;
    down_bb_[promote_id] = nullptr;

    fill_expert_buffers(gate_bb_[demote_id], up_bb_[demote_id], down_bb_[demote_id], gate, up, down, gate_scale,
                        up_scale, down_scale, full_intermediate);
  }

  /// \brief Fill three BufferBs from one expert's raw checkpoint bytes.
  ///
  /// Shared by the install and its bitwise verifier, deliberately: a check
  /// that reimplements what it checks verifies nothing.
  template <typename BB>
  void fill_expert_buffers(BB& gate_bb, BB& up_bb, BB& down_bb, intptr_t gate, intptr_t up, intptr_t down,
                           intptr_t gate_scale, intptr_t up_scale, intptr_t down_scale, int full_intermediate) {
    const int group_size = config_.quant_config.group_size;
    const size_t H = config_.hidden_size;
    const size_t I = config_.intermediate_size;
    const size_t wec = I * H;
    const size_t sec = (H / group_size) * I;
    const size_t i = (size_t)tp_part_idx;

    // from_raw_mat wants a contiguous source, and the staging arenas are
    // released after the bulk load, so slice into scratch. One expert of one
    // partition, a handful of times per swap window -- not worth an arena.
    std::vector<uint8_t> tmp(wec >> 1);

    // nth MUST match the bulk load's. from_raw_mat's destination offsets are
    // computed from n_block_size = (n_end - n_block_begin), so the packed AMX
    // layout is a FUNCTION OF nth: (ptr, 0, 1) covers the whole range but
    // arranges it differently than the union of (ptr, ith, nth). That was the
    // bug the bitwise check caught -- correct slices, wrong packing, and
    // nothing downstream notices.
    const int nth_gu = T::recommended_nth(config_.intermediate_size);
    const int nth_down = T::recommended_nth(config_.hidden_size);

    std::memcpy(tmp.data(), (const uint8_t*)gate + ((i * wec) >> 1), wec >> 1);
    for (int ith = 0; ith < nth_gu; ++ith) gate_bb->from_raw_mat(tmp.data(), ith, nth_gu);
    std::memcpy(tmp.data(), (const uint8_t*)up + ((i * wec) >> 1), wec >> 1);
    for (int ith = 0; ith < nth_gu; ++ith) up_bb->from_raw_mat(tmp.data(), ith, nth_gu);

    // down is [hidden, intermediate]: each column carries this partition's
    // slice of the full intermediate dimension.
    for (size_t col = 0; col < H; col++) {
      std::memcpy(tmp.data() + ((col * I) >> 1),
                  (const uint8_t*)down + (((col * (size_t)full_intermediate) + i * I) >> 1), I >> 1);
    }
    for (int ith = 0; ith < nth_down; ++ith) down_bb->from_raw_mat(tmp.data(), ith, nth_down);

    std::memcpy(gate_bb->d, (const uint8_t*)gate_scale + i * sec, sec);
    std::memcpy(up_bb->d, (const uint8_t*)up_scale + i * sec, sec);
    const size_t per_col = I / group_size;
    for (size_t col = 0; col < H; col++) {
      std::memcpy(down_bb->d + col * per_col,
                  (const uint8_t*)down_scale + (col * ((size_t)full_intermediate / group_size) + i * per_col), per_col);
    }
  }

  /// \brief Does the install reproduce, byte for byte, what the bulk load
  ///        produced for the SAME expert?
  ///
  /// The gate the demotion path actually needs. End-to-end quality can only
  /// say "something is worse"; this names it. A NUMA slice off by one
  /// partition writes a valid-looking expert, so nothing downstream fails --
  /// it just serves another partition's weights, which is exactly the class
  /// verify_row exists to catch on the GPU side.
  ///
  /// Runs on an expert that IS currently resident, filling scratch buffers
  /// through the same fill_expert_buffers the install uses, and comparing.
  /// Touches no live state.
  bool verify_install_against_loaded(int expert_id, intptr_t gate, intptr_t up, intptr_t down, intptr_t gate_scale,
                                     intptr_t up_scale, intptr_t down_scale, int full_intermediate) {
    if (gate_bb_[expert_id] == nullptr) throw std::runtime_error("verify: expert is not CPU-resident here");

    // this-> required: dependent base member, invisible to unqualified
    // two-phase lookup from the derived template.
    const size_t gu_bytes = this->buffer_b_required_size(config_.intermediate_size, config_.hidden_size);
    const size_t d_bytes = this->buffer_b_required_size(config_.hidden_size, config_.intermediate_size);

    void* g = std::aligned_alloc(64, gu_bytes);
    void* u = std::aligned_alloc(64, gu_bytes);
    void* d = std::aligned_alloc(64, d_bytes);
    auto gbb = this->make_buffer_b(config_.intermediate_size, config_.hidden_size, g);
    auto ubb = this->make_buffer_b(config_.intermediate_size, config_.hidden_size, u);
    auto dbb = this->make_buffer_b(config_.hidden_size, config_.intermediate_size, d);

    fill_expert_buffers(gbb, ubb, dbb, gate, up, down, gate_scale, up_scale, down_scale, full_intermediate);

    const bool ok = std::memcmp(gbb->b, gate_bb_[expert_id]->b, gu_bytes) == 0 &&
                    std::memcmp(ubb->b, up_bb_[expert_id]->b, gu_bytes) == 0 &&
                    std::memcmp(dbb->b, down_bb_[expert_id]->b, d_bytes) == 0;

    gbb.reset();
    ubb.reset();
    dbb.reset();
    std::free(g);
    std::free(u);
    std::free(d);
    return ok;
  }

  void load_weights() {
    auto& quant_config = config_.quant_config;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config_.physical_to_logical_map;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    if (quant_config.group_size != 32 || quant_config.zero_point)
      throw std::runtime_error("MXFP4 MoE requires group_size == 32 and no zero_point");
    if (config_.gate_scale == nullptr) throw std::runtime_error("MXFP4 MoE only support load native weight.");

    int nth = T::recommended_nth(config_.intermediate_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // Cold-only residency left this expert unallocated. Skipping is not
          // just null-safety: this loop is 75% of the 202 s startup, and
          // filling a buffer nothing reads is the bulk of it.
          if (gate_bb_[expert_idx] == nullptr) return;
          gate_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.gate_proj +
                  ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
          up_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.up_proj + ((logical_expert_id * config_.intermediate_size * config_.hidden_size) >> 1),
              ith, nth);
        },
        nullptr);

    nth = T::recommended_nth(config_.hidden_size);
    pool->do_work_stealing_job(
        nth * config_.expert_num, nullptr,
        [this, nth, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id / nth;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          int ith = task_id % nth;
          // Down is a SEPARATE work-stealing job from gate/up above -- three
          // loops in this function, not two. Missing this one segfaulted at
          // startup on the first cold-only run.
          if (down_bb_[expert_idx] == nullptr) return;
          down_bb_[expert_idx]->from_raw_mat(
              (uint8_t*)config_.down_proj +
                  ((logical_expert_id * config_.hidden_size * config_.intermediate_size) >> 1),
              ith, nth);
        },
        nullptr);

    pool->do_work_stealing_job(
        config_.expert_num, nullptr,
        [this, physical_to_logical_map](int task_id) {
          uint64_t expert_idx = task_id;
          uint64_t logical_expert_id = expert_map(physical_to_logical_map, expert_idx);
          size_t scale_elem_count = (config_.hidden_size * config_.intermediate_size) / config_.quant_config.group_size;
          // Unallocated under cold-only residency; see the weight loop above.
          if (gate_bb_[expert_idx] == nullptr) return;
          // E8M0 codes stay resident as raw bytes; the kernels expand them.
          std::memcpy(gate_bb_[expert_idx]->d, (const uint8_t*)config_.gate_scale + (logical_expert_id * scale_elem_count),
                      scale_elem_count);
          std::memcpy(up_bb_[expert_idx]->d, (const uint8_t*)config_.up_scale + (logical_expert_id * scale_elem_count),
                      scale_elem_count);
          std::memcpy(down_bb_[expert_idx]->d, (const uint8_t*)config_.down_scale + (logical_expert_id * scale_elem_count),
                      scale_elem_count);
        },
        nullptr);
  }

  static inline void fast_memcpy(void* __restrict dst, const void* __restrict src, size_t bytes) {
    uint8_t* d = (uint8_t*)dst;
    const uint8_t* s = (const uint8_t*)src;
    size_t chunks = bytes / 64;
    for (size_t i = 0; i < chunks; i++) {
      __m512i data = _mm512_loadu_si512((__m512i*)s);
      _mm512_storeu_si512((__m512i*)d, data);
      d += 64;
      s += 64;
    }
    if (bytes -= chunks * 64) std::memcpy(d, s, bytes);
  }

  // Resident scales are E8M0 bytes; the GPU-side scale buffer contract is bf16.
  // bf16 shares the fp32 exponent field, so the conversion is `code << 7` — the
  // same value the old fp32-resident path produced via fp32->bf16, exactly.
  static inline void fast_e8m0_to_bf16(ggml_bf16_t* __restrict dst, const uint8_t* __restrict src, size_t count) {
    size_t i = 0;
    for (; i + 32 <= count; i += 32) {
      __m256i codes = _mm256_loadu_si256((const __m256i*)(src + i));
      __m512i widened = _mm512_cvtepu8_epi16(codes);
      _mm512_storeu_si512((__m512i*)(dst + i), _mm512_slli_epi16(widened, 7));
    }
    for (; i < count; i++) dst[i] = e8m0_to_bf16(src[i]);
  }

  void write_weights_to_buffer(int gpu_tp_count, int cpu_tp_count, int expert_id, const GeneralMOEConfig& full_config,
                               const std::vector<uintptr_t>& w13_weight_ptrs,
                               const std::vector<uintptr_t>& w13_scale_ptrs,
                               const std::vector<uintptr_t>& w2_weight_ptrs,
                               const std::vector<uintptr_t>& w2_scale_ptrs) const {
    const int group_size = config_.quant_config.group_size;
    auto pool = config_.pool->get_subpool(tp_part_idx);

    size_t cpu_tp_weight_elem_count = (size_t)config_.intermediate_size * config_.hidden_size;
    size_t cpu_tp_weight_bytes = cpu_tp_weight_elem_count / 2;
    size_t cpu_tp_scale_elem_count = cpu_tp_weight_elem_count / group_size;

    size_t gpu_tp_weight_elem_count = (size_t)full_config.intermediate_size * full_config.hidden_size / gpu_tp_count;
    size_t gpu_tp_weight_bytes = gpu_tp_weight_elem_count / 2;
    size_t gpu_tp_scale_elem_count = gpu_tp_weight_elem_count / group_size;

    if (cpu_tp_count >= gpu_tp_count) {
      int target_gpu_tp = tp_part_idx / (cpu_tp_count / gpu_tp_count);
      int local_idx = tp_part_idx % (cpu_tp_count / gpu_tp_count);

      uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[target_gpu_tp];
      uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[target_gpu_tp];
      ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[target_gpu_tp];

      size_t offset_in_gpu_weight = local_idx * cpu_tp_weight_bytes;
      size_t offset_in_gpu_scale = local_idx * cpu_tp_scale_elem_count;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int total_tasks = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;

      size_t weight_chunk_size = (cpu_tp_weight_bytes + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, num_down_tasks, expert_id, weight_chunk_size, offset_in_gpu_weight, offset_in_gpu_scale,
           gpu_tp_weight_bytes, gpu_tp_scale_elem_count, w13_weight_dst, w13_scale_dst, w2_weight_dst, w2_scale_dst,
           group_size](int task_id) {
            if (task_id < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_id;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + start, (uint8_t*)gate_bb_[expert_id]->b + start,
                            end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, cpu_tp_weight_bytes);
              if (start < end)
                fast_memcpy(w13_weight_dst + offset_in_gpu_weight + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + start, end - start);
            } else if (task_id < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_id - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_col = config_.intermediate_size >> 1;
              size_t scale_per_col = config_.intermediate_size / group_size;
              size_t gpu_weight_stride = (full_config.intermediate_size / gpu_tp_count) >> 1;
              size_t gpu_scale_stride = (full_config.intermediate_size / gpu_tp_count) / group_size;
              size_t gpu_weight_slice_offset = local_idx * weight_per_col;
              size_t gpu_scale_slice_offset = local_idx * scale_per_col;

              for (size_t col = col_start; col < col_end; col++) {
                fast_memcpy(w2_weight_dst + col * gpu_weight_stride + gpu_weight_slice_offset,
                            (uint8_t*)down_bb_[expert_id]->b + col * weight_per_col, weight_per_col);
                fast_e8m0_to_bf16(w2_scale_dst + col * gpu_scale_stride + gpu_scale_slice_offset,
                                  down_bb_[expert_id]->d + col * scale_per_col, scale_per_col);
              }
            } else if (task_id == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_e8m0_to_bf16(w13_scale_dst + offset_in_gpu_scale, gate_bb_[expert_id]->d, cpu_tp_scale_elem_count);
            } else {
              fast_e8m0_to_bf16(w13_scale_dst + offset_in_gpu_scale + gpu_tp_scale_elem_count, up_bb_[expert_id]->d,
                                cpu_tp_scale_elem_count);
            }
          },
          nullptr);
    } else {
      int gpu_tps_per_cpu_tp = gpu_tp_count / cpu_tp_count;
      int start_gpu_tp = tp_part_idx * gpu_tps_per_cpu_tp;

      size_t data_per_gpu_tp_weight = cpu_tp_weight_bytes / gpu_tps_per_cpu_tp;
      size_t data_per_gpu_tp_scale = cpu_tp_scale_elem_count / gpu_tps_per_cpu_tp;

      constexpr int NUM_WEIGHT_TASKS = 8;
      constexpr int MIN_COLS_PER_TASK = 128;
      int num_down_tasks = std::max(1, (int)config_.hidden_size / MIN_COLS_PER_TASK);
      num_down_tasks = std::min(num_down_tasks, 32);
      int tasks_per_gpu_tp = NUM_WEIGHT_TASKS * 2 + num_down_tasks + 2;
      int total_tasks = tasks_per_gpu_tp * gpu_tps_per_cpu_tp;

      size_t weight_chunk_size = (data_per_gpu_tp_weight + NUM_WEIGHT_TASKS - 1) / NUM_WEIGHT_TASKS;
      weight_chunk_size = (weight_chunk_size + 63) & ~63ULL;

      pool->do_work_stealing_job(
          total_tasks, nullptr,
          [&, this, gpu_tps_per_cpu_tp, start_gpu_tp, data_per_gpu_tp_weight, data_per_gpu_tp_scale, num_down_tasks,
           tasks_per_gpu_tp, expert_id, weight_chunk_size, gpu_tp_weight_bytes, gpu_tp_scale_elem_count,
           group_size](int task_id) {
            int local_gpu_idx = task_id / tasks_per_gpu_tp;
            int task_type = task_id % tasks_per_gpu_tp;
            int gpu_tp_idx = start_gpu_tp + local_gpu_idx;

            uint8_t* w13_weight_dst = (uint8_t*)w13_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w13_scale_dst = (ggml_bf16_t*)w13_scale_ptrs[gpu_tp_idx];
            uint8_t* w2_weight_dst = (uint8_t*)w2_weight_ptrs[gpu_tp_idx];
            ggml_bf16_t* w2_scale_dst = (ggml_bf16_t*)w2_scale_ptrs[gpu_tp_idx];

            size_t cpu_offset_weight = local_gpu_idx * data_per_gpu_tp_weight;
            size_t cpu_offset_scale = local_gpu_idx * data_per_gpu_tp_scale;

            if (task_type < NUM_WEIGHT_TASKS) {
              int chunk_idx = task_type;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + start, (uint8_t*)gate_bb_[expert_id]->b + cpu_offset_weight + start,
                            end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS;
              size_t start = chunk_idx * weight_chunk_size;
              size_t end = std::min(start + weight_chunk_size, data_per_gpu_tp_weight);
              if (start < end)
                fast_memcpy(w13_weight_dst + gpu_tp_weight_bytes + start,
                            (uint8_t*)up_bb_[expert_id]->b + cpu_offset_weight + start, end - start);
            } else if (task_type < NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              int chunk_idx = task_type - NUM_WEIGHT_TASKS * 2;
              size_t cols_per_chunk = (config_.hidden_size + num_down_tasks - 1) / num_down_tasks;
              size_t col_start = chunk_idx * cols_per_chunk;
              size_t col_end = std::min(col_start + cols_per_chunk, (size_t)config_.hidden_size);

              size_t weight_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) >> 1;
              size_t scale_per_gpu_col = (config_.intermediate_size / gpu_tps_per_cpu_tp) / group_size;

              for (size_t col = col_start; col < col_end; col++) {
                size_t col_offset_weight = (col * config_.intermediate_size / 2) +
                                           (local_gpu_idx * data_per_gpu_tp_weight / config_.hidden_size);
                size_t col_offset_scale = (col * (config_.intermediate_size / group_size)) +
                                          (local_gpu_idx * data_per_gpu_tp_scale / config_.hidden_size);

                fast_memcpy(w2_weight_dst + col * weight_per_gpu_col,
                            (uint8_t*)down_bb_[expert_id]->b + col_offset_weight, weight_per_gpu_col);
                fast_e8m0_to_bf16(w2_scale_dst + col * scale_per_gpu_col, down_bb_[expert_id]->d + col_offset_scale,
                                  scale_per_gpu_col);
              }
            } else if (task_type == NUM_WEIGHT_TASKS * 2 + num_down_tasks) {
              fast_e8m0_to_bf16(w13_scale_dst, gate_bb_[expert_id]->d + cpu_offset_scale, data_per_gpu_tp_scale);
            } else {
              fast_e8m0_to_bf16(w13_scale_dst + gpu_tp_scale_elem_count, up_bb_[expert_id]->d + cpu_offset_scale,
                                data_per_gpu_tp_scale);
            }
          },
          nullptr);
    }
  }
};

// ============================================================================
// TP_MOE specialization for AMX_FP4_MOE_TP
// ============================================================================
template <typename K>
class TP_MOE<AMX_FP4_MOE_TP<K>> : public TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>> {
 public:
  using Base = TP_MOE<AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>>;
  using Base::Base;

  void load_weights() override {
    auto& config = this->config;
    auto& tps = this->tps;
    auto& tp_count = this->tp_count;
    auto pool = config.pool;
    const uint64_t* physical_to_logical_map = (const uint64_t*)config.physical_to_logical_map;

    bool use_per_expert_ptrs = !config.gate_projs.empty();

    if (config.gate_projs.empty() && config.gate_scale == nullptr)
      throw std::runtime_error("MXFP4 MoE only supports Packed FP4 with KGroup Scale");

    printf("From %s\n", use_per_expert_ptrs ? "per-expert pointers (gate_projs)" : "Packed FP4 with KGroup Scale");

    int& group_size = config.quant_config.group_size;

    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      size_t weight_elem_count = tpc.intermediate_size * tpc.hidden_size;
      size_t scales_elem_count = (tpc.hidden_size / group_size) * tpc.intermediate_size;

      const size_t w_bytes = (tpc.expert_num * weight_elem_count) / 2;
      const size_t s_bytes = tpc.expert_num * scales_elem_count;
      tpc.gate_proj = kt_staging_arena(i, 0, w_bytes);
      tpc.up_proj = kt_staging_arena(i, 1, w_bytes);
      tpc.down_proj = kt_staging_arena(i, 2, w_bytes);
      tpc.gate_scale = kt_staging_arena(i, 3, s_bytes);
      tpc.up_scale = kt_staging_arena(i, 4, s_bytes);
      tpc.down_scale = kt_staging_arena(i, 5, s_bytes);

      if (use_per_expert_ptrs) {
        pool->get_subpool(i)->do_work_stealing_job(
            tpc.expert_num, nullptr,
            [&, i](int expert_id_) {
              size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

              uint8_t* src_gate = (uint8_t*)config.gate_projs[0][expert_id];
              uint8_t* src_up = (uint8_t*)config.up_projs[0][expert_id];
              uint8_t* src_down = (uint8_t*)config.down_projs[0][expert_id];
              const uint8_t* src_gate_scale = (const uint8_t*)config.gate_scales[0][expert_id];
              const uint8_t* src_up_scale = (const uint8_t*)config.up_scales[0][expert_id];
              const uint8_t* src_down_scale = (const uint8_t*)config.down_scales[0][expert_id];

              memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                     src_gate + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                     src_up + ((i * weight_elem_count) >> 1), (weight_elem_count >> 1));
              memcpy((uint8_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                     src_gate_scale + (i * scales_elem_count), scales_elem_count);
              memcpy((uint8_t*)tpc.up_scale + (expert_id * scales_elem_count),
                     src_up_scale + (i * scales_elem_count), scales_elem_count);

              for (size_t col = 0; col < config.hidden_size; col++) {
                memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                       src_down + ((col * config.intermediate_size + i * tpc.intermediate_size) >> 1),
                       (tpc.intermediate_size >> 1));
                memcpy((uint8_t*)tpc.down_scale +
                           (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                       src_down_scale +
                           (col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                       (tpc.intermediate_size / group_size));
              }
            },
            nullptr);
      } else {
        if (tpc.load == false) {
          pool->get_subpool(i)->do_work_stealing_job(
              tpc.expert_num, nullptr,
              [&, i](int expert_id_) {
                size_t expert_id = expert_map(physical_to_logical_map, expert_id_);

                memcpy((uint8_t*)tpc.gate_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.gate_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.up_proj + ((expert_id * weight_elem_count) >> 1),
                       (uint8_t*)config.up_proj +
                           ((expert_id * config.intermediate_size * config.hidden_size + i * weight_elem_count) >> 1),
                       (weight_elem_count >> 1));
                memcpy((uint8_t*)tpc.gate_scale + (expert_id * scales_elem_count),
                       (const uint8_t*)config.gate_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       scales_elem_count);
                memcpy((uint8_t*)tpc.up_scale + (expert_id * scales_elem_count),
                       (const uint8_t*)config.up_scale +
                           (expert_id * (config.hidden_size / group_size) * config.intermediate_size +
                            i * scales_elem_count),
                       scales_elem_count);

                for (size_t col = 0; col < config.hidden_size; col++) {
                  memcpy((uint8_t*)tpc.down_proj + ((expert_id * weight_elem_count + col * tpc.intermediate_size) >> 1),
                         (uint8_t*)config.down_proj + ((expert_id * config.intermediate_size * config.hidden_size +
                                                        col * config.intermediate_size + i * tpc.intermediate_size) >>
                                                       1),
                         (tpc.intermediate_size >> 1));
                  memcpy((uint8_t*)tpc.down_scale +
                             (expert_id * scales_elem_count + col * (tpc.intermediate_size / group_size)),
                         (const uint8_t*)config.down_scale +
                             ((expert_id * (config.intermediate_size / group_size) * config.hidden_size) +
                              col * (config.intermediate_size / group_size) + i * (tpc.intermediate_size / group_size)),
                         (tpc.intermediate_size / group_size));
                }
              },
              nullptr);
        }
      }
      printf("TP %d load weight done.\n", i);
    });

    DO_TPS_LOAD_WEIGHTS(pool);

    // Staging is owned by the per-socket arenas and deliberately retained
    // for the next layer; clear the borrowed pointers so nothing mistakes
    // them for this layer's live weights (the residents were memcpy'd out
    // above by DO_TPS_LOAD_WEIGHTS).
    pool->dispense_backend()->do_numa_job([&, this](int i) {
      auto& tpc = tps[i]->config_;
      tpc.gate_proj = nullptr;
      tpc.up_proj = nullptr;
      tpc.down_proj = nullptr;
      tpc.gate_scale = nullptr;
      tpc.up_scale = nullptr;
      tpc.down_scale = nullptr;
    });

    this->weights_loaded = true;
  }

  /// \brief Move a promoted expert's CPU buffers to a demoted one and refill.
  ///
  /// The piece that lets cold-only residency and expert swapping run together.
  /// Without it a demotion points the forward at a null buffer, so the two
  /// features are refused in combination at config time.
  ///
  /// Neither allocates nor frees: swaps are 1:1, so the CPU-held count stays
  /// invariant and RSS stays flat. Runs on the CPUInfer TaskQueue, which the
  /// backend mutex makes mutually exclusive with poller forwards, and the swap
  /// window has already quiesced at a forward boundary.
  void swap_expert_slot(int promote_id, int demote_id, intptr_t gate, intptr_t up, intptr_t down,
                        intptr_t gate_scale, intptr_t up_scale, intptr_t down_scale) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    const int full_intermediate = this->config.intermediate_size;
    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->install_expert_from_raw(promote_id, demote_id, gate, up, down, gate_scale, up_scale, down_scale,
                                            full_intermediate);
    });
  }

  /// \brief Bitwise gate on the demotion install, across every partition.
  ///
  /// ANDs the per-partition results: a slice that is wrong on one NUMA node
  /// and right on the others is the most likely way to get this wrong, and
  /// would pass any check that looked at only one.
  bool verify_install_against_loaded(int expert_id, intptr_t gate, intptr_t up, intptr_t down, intptr_t gate_scale,
                                     intptr_t up_scale, intptr_t down_scale) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    const int full_intermediate = this->config.intermediate_size;
    std::atomic<int> bad{0};
    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      if (!this->tps[i]->verify_install_against_loaded(expert_id, gate, up, down, gate_scale, up_scale, down_scale,
                                                       full_intermediate))
        bad.fetch_add(1, std::memory_order_relaxed);
    });
    return bad.load(std::memory_order_relaxed) == 0;
  }

  // Per-expert addresses of the PERSISTENT CPU expert buffers (BufferB), one
  // row of six pointers per (NUMA partition, expert), so a caller can DMA the
  // bytes directly instead of asking kt to hand them over.
  //
  // WHY BufferB AND NOT tpc.gate_proj: tpc.gate_proj points at
  // kt_staging_arena, a scratch REUSED BY EVERY LAYER -- exporting it hands
  // out addresses that alias whichever layer loaded last (learned by
  // segfault, F2 2026-08-14). The persistent per-expert storage is gate_bb_/
  // up_bb_/down_bb_, and it holds checkpoint layout: from_raw_mat fills the
  // weight bytes with a plain row-major memcpy and scales_from_raw keeps the
  // raw E8M0 codes ("expanded at use time"). write_weights_to_buffer itself
  // reads these very buffers -- fast_memcpy for weights, fast_e8m0_to_bf16
  // for scales -- and that scale expansion is the ~38.7 ms of CPU work a
  // trtllm consumer has to undo, since trtllm wants the codes back as bytes.
  //
  // Returns (pointers, geometry):
  //   pointers[i * expert_num + e] =
  //       {gate_b, up_b, down_b, gate_d, up_d, down_d} for partition i,
  //       expert e -- six zeros when expert e is not CPU-resident here
  //       (cold-only leaves non-resident experts without buffers).
  //   geometry = {numa_count, expert_num, hidden_size,
  //               intermediate_size_per_partition, group_size}
  //
  // Layout per entry: gate/up are [intermediate_per_partition, hidden/2]
  // bytes row-major; down is [hidden, intermediate_per_partition/2] --
  // kt's usual column blocking. Scales are raw E8M0, one code per group_size
  // values, same orientation as their weights.
  //
  // Pointers are stable for the life of this object EXCEPT across
  // swap_expert_slot, which moves buffer ownership between expert ids; a
  // caller that swaps must re-fetch. The memory is NOT pinned; DMA requires
  // registering it.
  std::pair<std::vector<std::vector<uintptr_t>>, std::vector<int64_t>> expert_buffer_pointers() const {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");

    const auto& tpc0 = this->tps[0]->config_;
    const int64_t expert_num = (int64_t)tpc0.expert_num;

    std::vector<std::vector<uintptr_t>> ptrs;
    ptrs.reserve(this->tps.size() * expert_num);
    for (size_t i = 0; i < this->tps.size(); i++) {
      // Through the base, where gate_bb_ et al. are public; AMX_FP4_MOE_TP's
      // `using Base::gate_bb_` re-declarations sit in its private section.
      const auto& tp = static_cast<const AMX_MOE_BASE<K, AMX_FP4_MOE_TP<K>>&>(*this->tps[i]);
      for (int64_t e = 0; e < expert_num; e++) {
        if (tp.gate_bb_[e] == nullptr || tp.up_bb_[e] == nullptr || tp.down_bb_[e] == nullptr) {
          ptrs.push_back({0, 0, 0, 0, 0, 0});
          continue;
        }
        ptrs.push_back({(uintptr_t)tp.gate_bb_[e]->b, (uintptr_t)tp.up_bb_[e]->b, (uintptr_t)tp.down_bb_[e]->b,
                        (uintptr_t)tp.gate_bb_[e]->d, (uintptr_t)tp.up_bb_[e]->d, (uintptr_t)tp.down_bb_[e]->d});
      }
    }
    std::vector<int64_t> geometry = {(int64_t)this->tps.size(), expert_num, (int64_t)tpc0.hidden_size,
                                     (int64_t)tpc0.intermediate_size, (int64_t)this->config.quant_config.group_size};
    return {ptrs, geometry};
  }

  void write_weight_scale_to_buffer(int gpu_tp_count, int expert_id, const std::vector<uintptr_t>& w13_weight_ptrs,
                                    const std::vector<uintptr_t>& w13_scale_ptrs,
                                    const std::vector<uintptr_t>& w2_weight_ptrs,
                                    const std::vector<uintptr_t>& w2_scale_ptrs) {
    if (!this->weights_loaded) throw std::runtime_error("Not Loaded");
    if (this->tps.empty()) throw std::runtime_error("No TP parts initialized");
    if (w13_weight_ptrs.size() != gpu_tp_count || w13_scale_ptrs.size() != gpu_tp_count ||
        w2_weight_ptrs.size() != gpu_tp_count || w2_scale_ptrs.size() != gpu_tp_count)
      throw std::runtime_error("Pointer arrays size must match gpu_tp_count");

    this->config.pool->dispense_backend()->do_numa_job([&, this](int i) {
      this->tps[i]->write_weights_to_buffer(gpu_tp_count, this->tp_count, expert_id, this->config, w13_weight_ptrs,
                                            w13_scale_ptrs, w2_weight_ptrs, w2_scale_ptrs);
    });
  }
};

#endif  // CPUINFER_OPERATOR_AMX_FP4_MOE_H
