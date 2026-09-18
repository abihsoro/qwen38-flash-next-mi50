// n_e1_hbm_ceiling.cu — N-E1: HBM streaming ceiling under real access
// patterns, post-soak, at the H1 (150 W) cap. No datasheet figures.
// Patterns: (a) pure read stream (weight-streaming shape), (b) copy
// (read+write), (c) row-gather with gaps (n-gram table pattern at scale).
// Protocol: >= 15 repeats, median + IQR reported (work order I5-style).
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void read_k(const float* __restrict__ a, float* __restrict__ out,
                       int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[threadIdx.x] = a[i];  // tiny write; read-bound
}
__global__ void copy_k(const float* __restrict__ a, float* __restrict__ b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) b[i] = a[i];
}
// row-gather: read ROWS of ROW_WORDS floats strided by STRIDE (n-gram-ish)
__global__ void gather_k(const float* __restrict__ table, const int* __restrict__ idx,
                         float* __restrict__ out, int rows, int row_words, int stride) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * row_words;
  for (int i = t; i < total; i += gridDim.x * blockDim.x) {
    int r = i / row_words, c = i % row_words;
    out[i] = table[(size_t)idx[r] * stride + c];
  }
}

static double median_iqr(std::vector<double>& v, double* iqr) {
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  double med = v[n / 2];
  *iqr = v[(3 * n) / 4] - v[n / 4];
  return med;
}

int main() {
  const int n = 512 << 20;  // 2 GiB float
  const int reps = 20;
  float *a, *b, *out;
  hipMalloc(&a, n * 4);
  hipMalloc(&b, n * 4);
  hipMalloc(&out, 256 * 4);
  hipMemset(a, 1, n * 4);
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto bench = [&](auto launch, double bytes, const char* name) {
    std::vector<double> us;
    for (int r = 0; r < reps; r++) {
      hipEventRecord(e0);
      launch();
      hipEventRecord(e1);
      hipEventSynchronize(e1);
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      us.push_back(ms * 1000.0);
    }
    double iqr;
    double med = median_iqr(us, &iqr);
    printf("%-28s median %7.1f us  IQR %6.1f  %6.1f GB/s\n", name, med, iqr,
           bytes / (med * 1e-6) / 1e9);
  };
  const int blocks = (n + 255) / 256;
  bench([&] { read_k<<<blocks, 256>>>(a, out, n); }, (double)n * 4,
        "read-stream (2 GiB)");
  bench([&] { copy_k<<<blocks, 256>>>(a, b, n); }, (double)n * 4 * 2,
        "copy (2 GiB r+w)");
  // row-gather: 256 KiB rows x 5 KiB, stride 5 KiB (n-gram pattern), 1 GiB read
  {
    const int rows = 1 << 18;
    const int row_words = 1280;  // 5 KiB in floats
    const int stride = row_words;
    int* idx;
    float* gtab;
    float* gout;
    hipMalloc(&idx, rows * 4);
    hipMalloc(&gtab, (size_t)rows * stride * 4);
    hipMalloc(&gout, (size_t)rows * row_words * 4);
    hipMemset(gtab, 1, (size_t)rows * stride * 4);
    std::vector<int> hidx(rows);
    for (int i = 0; i < rows; i++) hidx[i] = (i * 7919) % rows;
    hipMemcpy(idx, hidx.data(), rows * 4, hipMemcpyHostToDevice);
    auto g = [&] { gather_k<<<256, 256>>>(gtab, idx, gout, rows, row_words, stride); };
    bench(g, (double)rows * row_words * 4, "row-gather 5KiB rows (1 GiB)");
  }
  printf("N-E1 done at 150 W cap, post 600s soak\n");
  return 0;
}
