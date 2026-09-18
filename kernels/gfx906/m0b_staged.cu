// m0b_staged.cu — guaranteed cross-device path: D2H into pinned host + H2D out,
// plus same-device D2D reference. Strict device discipline, no peer machinery.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, \
          hipGetErrorString(e)); exit(1); } } while (0)

static double med(std::vector<double>& v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }

int main(int argc, char** argv){
  int nd=0; CK(hipGetDeviceCount(&nd));
  if (nd < 2) return 2;
  size_t bytes = 64ull<<20; int reps = argc>1?atoi(argv[<bus>]):10;
  char* a=nullptr; char* b=nullptr; char* h=nullptr;
  CK(hipSetDevice(0)); CK(hipMalloc(&a, bytes)); CK(hipMemset(a, 0xAB, bytes));
  CK(hipSetDevice(1)); CK(hipMalloc(&b, bytes)); CK(hipMemset(b, 0x00, bytes));
  CK(hipHostMalloc(&h, bytes));
  CK(hipDeviceSynchronize());

  std::vector<double> s01;
  for (int r=0;r<reps;r++){
    hipEvent_t ea,eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
    CK(hipSetDevice(0)); CK(hipEventRecord(ea));
    CK(hipMemcpy(h, a, bytes, hipMemcpyDeviceToHost));
    CK(hipSetDevice(1));
    CK(hipMemcpy(b, h, bytes, hipMemcpyHostToDevice));
    CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
    float ms=0; CK(hipEventElapsedTime(&ms,ea,eb));
    s01.push_back((double)bytes/(ms*1e-3)/1e9);
    CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
  }
  printf("staged D2H+H2D 0->1: %.2f GB/s median\n", med(s01)); fflush(stdout);

  char* a2=nullptr; CK(hipSetDevice(0)); CK(hipMalloc(&a2, bytes));
  std::vector<double> d0;
  for (int r=0;r<reps;r++){
    hipEvent_t ea,eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
    CK(hipEventRecord(ea));
    CK(hipMemcpy(a2, a, bytes, hipMemcpyDeviceToDevice));
    CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
    float ms=0; CK(hipEventElapsedTime(&ms,ea,eb));
    d0.push_back((double)bytes/(ms*1e-3)/1e9);
    CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
  }
  printf("same-device D2D on card0: %.2f GB/s median\n", med(d0)); fflush(stdout);
  CK(hipFree(a2));
  CK(hipSetDevice(0)); CK(hipFree(a));
  CK(hipSetDevice(1)); CK(hipFree(b));
  CK(hipHostFree(h));
  printf("m0b done\n");
  return 0;
}
