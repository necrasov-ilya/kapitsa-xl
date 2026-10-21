// Checks the bounded per-slot down projection against a CPU reference and the original staged kernel.
#include "metal/gpu.h"
#include "bench/bench.h"
#include "util/bf16.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using namespace ni;

struct Params { uint32_t D, I, SI, k, shared_bits; };
struct Matrix { Buffer w, s, b; uint32_t N, K; };
static Matrix matrix(uint32_t n, uint32_t k) {
  auto& g = Gpu::get();
  Matrix m{g.alloc(size_t(n) * k, "weights"), g.alloc(size_t(n) * k / 64 * 2, "scales"),
           g.alloc(size_t(n) * k / 64 * 2, "biases"), n, k};
  for (size_t j = 0; j < size_t(n) * k; ++j) m.w.as<uint8_t>()[j] = uint8_t((j * 73 + j / k * 17) % 251);
  for (size_t j = 0; j < size_t(n) * k / 64; ++j) {
    m.s.as<uint16_t>()[j] = f32_to_bf16(0.0078125f);
    m.b.as<uint16_t>()[j] = f32_to_bf16(-1.0f);
  }
  return m;
}
static int check(uint32_t width) {
  auto& g = Gpu::get();
  constexpr uint32_t D = 128, E = 16, k = 8, R = 2;
  auto routed = matrix(E * D, width), shared = matrix(D, width);
  auto h = g.alloc(size_t(R) * (k + 1) * width * 4, "activations");
  auto ids = g.alloc(R * k * 4, "ids"), weights = g.alloc(R * (k + 1) * 4, "router");
  auto x = g.alloc(R * D * 4, "output");
  for (uint32_t j = 0; j < R * (k + 1) * width; ++j) h.as<float>()[j] = float(int(j % 37) - 18) / 32;
  for (uint32_t j = 0; j < R * k; ++j) ids.as<uint32_t>()[j] = (j * 3) % E;
  for (uint32_t j = 0; j < R * (k + 1); ++j) weights.as<float>()[j] = float(j % (k + 1) + 1) / 64;
  auto run = [&](const char* name) {
    memset(x.ptr(), 0, x.size);
    auto e = g.begin();
    e->use(g.pipeline(name));
    e->buf(0, routed.w); e->buf(1, routed.s); e->buf(2, routed.b);
    e->buf(3, shared.w); e->buf(4, shared.s); e->buf(5, shared.b);
    e->buf(6, h); e->buf(7, ids); e->buf(8, weights); e->buf(9, x);
    Params p{D, width, width, k, 8}; e->val(10, p);
    e->dispatch(D / 16, R, 1, 256);
    e->commit_and_wait();
  };
  run("moeg_down_rows_f8g64_f8g64_slot");
  std::vector<float> got(x.as<float>(), x.as<float>() + R * D);
  double err = 0, scale = 0;
  for (uint32_t r = 0; r < R; ++r) for (uint32_t n = 0; n < D; ++n) {
    double sum = 0;
    for (uint32_t slot = 0; slot <= k; ++slot) {
      const Matrix& m = slot == k ? shared : routed;
      uint32_t row = slot == k ? n : ids.as<uint32_t>()[r * k + slot] * D + n;
      double dot = 0;
      for (uint32_t j = 0; j < width; ++j)
        dot += (double(m.w.as<uint8_t>()[size_t(row) * width + j]) / 128 - 1) *
               h.as<float>()[(r * (k + 1) + slot) * width + j];
      sum += weights.as<float>()[r * (k + 1) + slot] * dot;
    }
    err = std::max(err, std::abs(got[r * D + n] - sum)); scale = std::max(scale, std::abs(sum));
  }
  if (width == 512 && !getenv("MTL_SHADER_VALIDATION")) {
    run("moeg_down_rows_f8g64_f8g64");
    if (memcmp(got.data(), x.ptr(), got.size() * sizeof(float))) {
      fprintf(stderr, "512-wide slot and staged outputs differ\n"); return 1;
    }
  }
  printf("width %u, %u staged values, relative error %.3g\n", width, (k + 1) * width, err / std::max(1.0, scale));
  return err > 1e-5 * std::max(1.0, scale);
}
int main() {
  ni::bench::GpuLock lock;
  if (!getenv("NI_LOCK_HELD") && !getenv("NACE_GPU_LOCK_HELD") && !lock.acquire()) {
    perror("GPU lock");
    return 1;
  }
  return check(512) | check(1024);
}
