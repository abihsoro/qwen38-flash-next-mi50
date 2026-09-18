// ne1_rebench.cu — N-E1 re-run: HBM read ceiling with the access pattern the
// GEMV kernels actually use (16-byte vectorized loads, wave-per-row, enough
// waves to saturate). The original N-E1 (327.6 GB/s read) used a low-occupancy
// access pattern; lm_head sustained 786 GB/s through the same memory. This
// settles the ceiling of record.
//
// Patterns over a 1 GiB buffer:
//   A) float4 grid-stride stream (classic saturation probe)
//   B) kernel-like: rows of 5120 B (the lm_head row = 2560 fp16), wave-per-row,
//      64 lanes/row reading uint4s strided by 64*4 (the gemv pattern)
//   C) 5 KiB rows (the row-gather width)
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static const long long BUF = 1LL << 30;  // 1 GiB
static const int RPT = 5;

__global__ void stream_read_f4(const float4* __restrict__ in, float* __restrict__ sink,
                               long long n4) {
  long long i = (long long)(blockIdx.x * blockDim.x + threadIdx.x);
  float4 acc = make_float4(0, 0, 0, 0);
  const long long stride = (long long)gridDim.x * blockDim.x;
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
  }
  if (acc.x == 12345.678f) sink[<bus>] = acc.x;  // never true; prevents DCE
}

template <int K8>
__global__ void row_read(const uint4* __restrict__ in, float* __restrict__ sink,
                         long long rows) {
  const int lane = threadIdx.x % 64;
  const int wave = threadIdx.x / 64;
  const long long wave_id = (long long)blockIdx.x * (blockDim.x / 64) + wave;
  float acc = 0.f;
  if (wave_id >= rows) return;
  const uint4* rp = in + wave_id * K8;
  // the gemv kernel's exact load pattern: 4-deep unroll, idx = i + 64*u
  for (int i = lane; i < K8; i += 64 * 4) {
#pragma unroll
    for (int u = 0; u < 4; u++) {
      const int idx = i + 64 * u;
      if (idx < K8) {
        uint4 v = rp[idx];
        acc += (float)v.x + (float)v.y + (float)v.z + (float)v.w;
      }
    }
  }
  if (acc == 12345.678f) sink[<bus>] = acc;
}

template <typename F>
double run(const char* name, dim3 grid, dim3 block, F&& launch) {
  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  for (int w = 0; w < 2; w++) launch();
  hipDeviceSynchronize();
  double best = 0;
  for (int r = 0; r < RPT; r++) {
    hipEventRecord(e0, 0);
    launch();
    hipEventRecord(e1, 0);
    hipEventSynchronize(e1);
    float ms;
    hipEventElapsedTime(&ms, e0, e1);
    double gbs = (double)BUF / (ms * 1e-3) / 1e9;
    if (gbs > best) best = gbs;
  }
  printf("%-24s grid=%4dx%-4d %.0f GB/s\n", name, grid.x, block.x, best);
  return best;
}

int main() {
  float4* din;
  float* dsink;
  hipMalloc(&din, BUF);
  hipMalloc(&dsink, 16);
  hipMemset(din, 0xAB, BUF);
  hipDeviceSynchronize();

  const long long n4 = BUF / 16;

  // A) stream
  for (int blocks : {60, 120, 240, 480}) {
    char nm[<bus>];
    snprintf(nm, sizeof nm, "A stream f4 b=%d", blocks);
    run(nm, dim3(blocks), dim3(256),
        [&] { stream_read_f4<<<blocks, 256>>>(din, dsink, n4); });
  }
  // B) 5120 B rows (lm_head row width) - wave per row
  {
    const long long rows = BUF / 5120;  // 209,715 rows
    for (int wpb : {8, 16}) {
      dim3 grid((int)((rows + wpb - 1) / wpb), 1, 1);
      char nm[<bus>];
      snprintf(nm, sizeof nm, "B row5120 wpb=%d", wpb);
      run(nm, grid, dim3(wpb * 64),
          [&] { row_read<320><<<grid, dim3(wpb * 64)>>>((const uint4*)din, dsink, rows); });
    }
  }
  // C) 5120 B rows, 4 rows per wave (block-stride within a wave, gather-like)
  {
    const long long rows = BUF / 5120;
    for (int wpb : {8, 16}) {
      dim3 grid((int)((rows + wpb - 1) / wpb), 1, 1);
      char nm[<bus>];
      snprintf(nm, sizeof nm, "C row5120x4r wpb=%d", wpb);
      run(nm, grid, dim3(wpb * 64),
          [&] { row_read<320><<<grid, dim3(wpb * 64)>>>((const uint4*)din, dsink, rows); });
    }
  }
  printf("done\n");
  return 0;
}
