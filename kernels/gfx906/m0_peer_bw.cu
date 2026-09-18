// m0_peer_bw.cu — two-card P2P determination (G3 / M0 pre-check, D103 window).
// Strict device discipline: every pointer is memset/copied only with its own
// device current. Peer sections are GUARDED on enable success — a peer
// failure is reported, not crashed on. Driver-path hipMemcpyPeer measured
// regardless (it may work via bounce or error).
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, \
          hipGetErrorString(e)); exit(1); } } while (0)

__global__ void peer_writer(float4* __restrict__ dst, size_t n4) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float4 v = make_float4(1.f, 2.f, 3.f, 4.f);
  for (; i < n4; i += stride) dst[i] = v;
}

__global__ void peer_reader(const float4* __restrict__ src, size_t n4,
                            float* __restrict__ out) {
  size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n4; i += stride) { float4 v = src[i]; acc += v.x + v.y + v.z + v.w; }
  if (acc == 12345.678f) out[blockIdx.x] = acc;
}

static double median(std::vector<double>& v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

int main(int argc, char** argv) {
  int ndev = 0; CK(hipGetDeviceCount(&ndev));
  printf("m0_peer_bw: %d device(s)\n", ndev); fflush(stdout);
  if (ndev < 2) { printf("SKIP: requires 2 GPUs\n"); return 2; }

  for (int d = 0; d < ndev; d++) {
    hipDeviceProp_t p; CK(hipGetDeviceProperties(&p, d));
    printf("  device %d: %s  pci %02x:%02x.%x  vram %zu MB\n", d, p.name,
           p.pciBusID, p.pciDeviceID, p.pciDomainID, (size_t)(p.totalGlobalMem >> 20));
  }
  fflush(stdout);

  printf("\n[<bus>] hipDeviceCanAccessPeer matrix\n");
  int can[<bus>][<bus>] = {{0, 0}, {0, 0}};
  for (int a = 0; a < ndev; a++)
    for (int b = 0; b < ndev; b++) {
      hipDeviceCanAccessPeer(&can[<bus>][<bus>], a, b);
      printf("    %d->%d: %d\n", a, b, can[<bus>][<bus>]);
    }
  fflush(stdout);

  printf("\n[<bus>] hipDeviceEnablePeerAccess\n");
  bool en01 = false, en10 = false;
  CK(hipSetDevice(0));
  hipError_t e01 = hipDeviceEnablePeerAccess(1, 0);
  printf("    0->1: %s (%s)\n", e01 == hipSuccess ? "OK" : "FAIL",
         hipGetErrorString(e01));
  en01 = (e01 == hipSuccess);
  CK(hipSetDevice(1));
  hipError_t e10 = hipDeviceEnablePeerAccess(0, 0);
  printf("    1->0: %s (%s)\n", e10 == hipSuccess ? "OK" : "FAIL",
         hipGetErrorString(e10));
  en10 = (e10 == hipSuccess);
  fflush(stdout);
  bool peer_ok = en01 && en10;
  printf("  kernel-level P2P available: %s\n", peer_ok ? "YES" : "NO (guarded out)");

  static const size_t kSizes[] = {16ull << 20, 64ull << 20, 256ull << 20};
  const int reps = (argc > 1) ? atoi(argv[<bus>]) : 10;
  const int blocks = 4096, threads = 256;

  for (size_t bytes : kSizes) {
    size_t n4 = bytes / 16;
    printf("\n[sizes %zu MB, reps %d]\n", bytes >> 20, reps);
    fflush(stdout);

    float4* b0 = nullptr; float4* b1 = nullptr;
    CK(hipSetDevice(0)); CK(hipMalloc(&b0, bytes)); CK(hipMemset(b0, 0, bytes));
    CK(hipSetDevice(1)); CK(hipMalloc(&b1, bytes)); CK(hipMemset(b1, 0, bytes));
    float* sink0 = nullptr; float* sink1 = nullptr;
    CK(hipSetDevice(0)); CK(hipMalloc(&sink0, blocks * sizeof(float)));
    CK(hipSetDevice(1)); CK(hipMalloc(&sink1, blocks * sizeof(float)));
    CK(hipDeviceSynchronize());

    if (peer_ok) {
      // kernel WRITE 0->1
      std::vector<double> tw;
      CK(hipSetDevice(0));
      for (int r = 0; r < reps; r++) {
        hipEvent_t ea, eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
        CK(hipEventRecord(ea));
        peer_writer<<<blocks, threads>>>(b1, n4);
        CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
        float ms = 0; CK(hipEventElapsedTime(&ms, ea, eb));
        tw.push_back((double)bytes / (ms * 1e-3) / 1e9);
        CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
      }
      float4 probe[<bus>];
      CK(hipSetDevice(1));
      CK(hipMemcpy(probe, b1, sizeof(probe), hipMemcpyDeviceToHost));
      int ok_w = (probe[<bus>].x == 1.f && probe[<bus>].w == 4.f) ? 1 : 0;
      printf("  kernel WRITE 0->1: %.2f GB/s median (validated %s)\n",
             median(tw), ok_w ? "ok" : "MISMATCH");

      // kernel READ 0->1
      std::vector<double> tr;
      CK(hipSetDevice(0));
      for (int r = 0; r < reps; r++) {
        hipEvent_t ea, eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
        CK(hipEventRecord(ea));
        peer_reader<<<blocks, threads>>>(b1, n4, sink0);
        CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
        float ms = 0; CK(hipEventElapsedTime(&ms, ea, eb));
        tr.push_back((double)bytes / (ms * 1e-3) / 1e9);
        CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
      }
      printf("  kernel READ  0->1: %.2f GB/s median\n", median(tr));
      fflush(stdout);
    } else {
      printf("  kernel WRITE/READ skipped (no peer enable)\n");
    }

    // driver-path hipMemcpyPeer (may bounce or fail; never crash)
    std::vector<double> tc01, tc10;
    bool mc01_ok = true, mc10_ok = true;
    CK(hipSetDevice(0));
    for (int r = 0; r < reps; r++) {
      hipEvent_t ea, eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
      CK(hipEventRecord(ea));
      hipError_t e = hipMemcpyPeer(b1, 1, b0, 0, bytes);
      if (e != hipSuccess) { printf("  memcpyPeer 0->1 FAILED: %s\n", hipGetErrorString(e)); mc01_ok = false; break; }
      CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
      float ms = 0; CK(hipEventElapsedTime(&ms, ea, eb));
      tc01.push_back((double)bytes / (ms * 1e-3) / 1e9);
      CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
    }
    if (mc01_ok) printf("  memcpyPeer   0->1: %.2f GB/s median\n", median(tc01));
    CK(hipSetDevice(1));
    for (int r = 0; r < reps; r++) {
      hipEvent_t ea, eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
      CK(hipEventRecord(ea));
      hipError_t e = hipMemcpyPeer(b0, 0, b1, 1, bytes);
      if (e != hipSuccess) { printf("  memcpyPeer 1->0 FAILED: %s\n", hipGetErrorString(e)); mc10_ok = false; break; }
      CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
      float ms = 0; CK(hipEventElapsedTime(&ms, ea, eb));
      tc10.push_back((double)bytes / (ms * 1e-3) / 1e9);
      CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
    }
    if (mc10_ok) printf("  memcpyPeer   1->0: %.2f GB/s median\n", median(tc10));
    fflush(stdout);

    CK(hipSetDevice(0)); CK(hipFree(b0)); CK(hipFree(sink0));
    CK(hipSetDevice(1)); CK(hipFree(b1)); CK(hipFree(sink1));
  }
  printf("\ndone.\n");
  return 0;
}
