// health.cpp — minimal per-device HIP health probe (non-destructive).
// Answers: is every GPU still usable after the ACS negative-control incident?
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

int main() {
  int n = 0;
  hipError_t e = hipGetDeviceCount(&n);
  printf("hipGetDeviceCount -> %s (n=%d)\n", hipGetErrorString(e), n);
  if (e != hipSuccess) return 2;

  int bad = 0;
  for (int i = 0; i < n; i++) {
    hipDeviceProp_t p;
    hipError_t pe = hipGetDeviceProperties(&p, i);
    if (pe != hipSuccess) { printf("dev%d props FAILED: %s\n", i, hipGetErrorString(pe)); bad++; continue; }
    hipError_t se = hipSetDevice(i);
    if (se != hipSuccess) { printf("dev%d setDevice FAILED: %s\n", i, hipGetErrorString(se)); bad++; continue; }

    void* d = nullptr;
    hipError_t me = hipMalloc(&d, 1u << 20);
    if (me != hipSuccess) { printf("dev%d malloc FAILED: %s\n", i, hipGetErrorString(me)); bad++; continue; }

    hipError_t mse = hipMemset(d, 0x3C, 1u << 20);
    unsigned char* h = (unsigned char*)malloc(1u << 20);
    hipError_t ce = hipMemcpy(h, d, 1u << 20, hipMemcpyDeviceToHost);
    hipError_t ye = hipDeviceSynchronize();
    int ok = 1;
    if (ce == hipSuccess) {
      for (int k = 0; k < (1 << 20); k++) if (h[k] != 0x3C) { ok = 0; break; }
    } else {
      ok = 0;
    }
    printf("dev%d %-22s %-22s memset=%-14s memcpy=%-14s sync=%-14s verify=%s\n",
           i, p.name, p.gcnArchName, hipGetErrorString(mse), hipGetErrorString(ce),
           hipGetErrorString(ye), ok ? "OK" : "FAIL");
    if (!ok || mse != hipSuccess || ye != hipSuccess) bad++;
    free(h);
    hipFree(d);
  }
  printf("HEALTH: %d/%d devices fully functional\n", n - bad, n);
  return bad ? 1 : 0;
}
