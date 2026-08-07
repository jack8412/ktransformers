// Bitwise exactness test for the MXFP4 E2M1 -> BF16 PSHUFB decode used by
// GemmKernel224MXFP4SmallKGroup. Exhaustively sweeps every packed byte value
// in every lane position and compares against an independent scalar reference.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "../fp4-moe.hpp"

namespace {

// Independent scalar reference for the 16 E2M1 code points (bit 3 = sign).
const float kE2M1[16] = {0.0f, 0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

uint16_t bf16_bits(float v) {
  uint32_t u;
  std::memcpy(&u, &v, 4);
  // Truncation == rounding here: E2M1 values carry <= 3 significand bits.
  return (uint16_t)(u >> 16);
}

}  // namespace

int main() {
  using K = amx::GemmKernel224MXFP4SmallKGroup;
  alignas(64) uint8_t in[16];
  alignas(64) uint16_t out[32];

  std::mt19937 rng(42);
  long checked = 0;

  // Every byte value in every lane position (256 * 16 patterns), with the
  // other 15 lanes randomized each time.
  for (int byte = 0; byte < 256; byte++) {
    for (int lane = 0; lane < 16; lane++) {
      for (int j = 0; j < 16; j++) in[j] = (uint8_t)(rng() & 0xFF);
      in[lane] = (uint8_t)byte;

      __m512i v = K::mxfp4_to_bf16_32(_mm_load_si128((const __m128i*)in));
      _mm512_storeu_si512((__m512i*)out, v);

      // Column order per mxfp4_to_bf16_32: out[2j] = lo nibble (element 2j),
      // out[2j+1] = hi nibble (element 2j+1).
      for (int j = 0; j < 16; j++) {
        uint16_t lo_ref = bf16_bits(kE2M1[in[j] & 0x0F]);
        uint16_t hi_ref = bf16_bits(kE2M1[(in[j] >> 4) & 0x0F]);
        if (out[2 * j] != lo_ref || out[2 * j + 1] != hi_ref) {
          printf("FAIL byte=0x%02x lane=%d j=%d: got (0x%04x, 0x%04x) want (0x%04x, 0x%04x)\n", byte, lane, j,
                 out[2 * j], out[2 * j + 1], lo_ref, hi_ref);
          return 1;
        }
        checked += 2;
      }
    }
  }
  printf("PASS: mxfp4_to_bf16_32 bitwise-exact over %ld decoded values\n", checked);

  // --- E8M0 scale expansion: all 256 codes ---
  // The resident layout may store scales as fp32 (widened at load) or as raw
  // E8M0 bytes expanded at use; either way the fp32 value a k-group is scaled
  // by must be identical. Pin every route against each other and against ldexp,
  // so a future storage change cannot silently rescale whole k-groups.
  int scale_bad = 0;
  for (int code = 0; code < 256; code++) {
    float got = e8m0_to_fp32((uint8_t)code);
    uint32_t got_bits;
    std::memcpy(&got_bits, &got, 4);

    // Load-time route: bf16 bits (code << 7) widened to fp32 by << 16.
    uint32_t via_bf16_bits = (uint32_t)((uint16_t)code << 7) << 16;
    // Independent reference: 2^(code-127) for normal codes.
    uint32_t ref_bits;
    if (code == 0) {
      ref_bits = 0;  // 2^-127 is below fp32 normals; the repo flushes to +0.0
    } else {
      float ref = std::ldexp(1.0f, code - 127);
      std::memcpy(&ref_bits, &ref, 4);
    }
    uint32_t bf_widened = (uint32_t)e8m0_to_bf16((uint8_t)code).bits << 16;

    if (got_bits != via_bf16_bits || got_bits != ref_bits || bf_widened != got_bits) {
      printf("FAIL e8m0 code=%d: use-time=0x%08x load-time=0x%08x ref=0x%08x bf16<<16=0x%08x\n", code, got_bits,
             via_bf16_bits, ref_bits, bf_widened);
      if (++scale_bad > 4) return 1;
    }
  }
  if (scale_bad) return 1;
  printf("PASS: e8m0_to_fp32/e8m0_to_bf16 exact for all 256 codes (code 0 -> +0.0, 255 -> +inf)\n");
  return 0;
}
