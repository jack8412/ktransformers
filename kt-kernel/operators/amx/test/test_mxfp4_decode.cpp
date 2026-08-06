// Bitwise exactness test for the MXFP4 E2M1 -> BF16 PSHUFB decode used by
// GemmKernel224MXFP4SmallKGroup. Exhaustively sweeps every packed byte value
// in every lane position and compares against an independent scalar reference.
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
  return 0;
}
