// m0a_memcpy_peer.cu — hipMemcpyPeer 0<->1 behavior when canAccessPeer may be
// 0. Strict device discipline. Does NOT abort on peer-enable failure.
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
  printf("m0a: %d devices\n", nd); fflush(stdout);
  if (nd < 2) return 2;
  size_t bytes = 64ull<<20; int reps = argc>1?atoi(argv[<bus>]):10;
  int can01=0, can10=0;
  hipDeviceCanAccessPeer(&can01, 0, 1);
  hipDeviceCanAccessPeer(&can10, 1, 0);
  printf("canAccessPeer 0->1 = %d   1->0 = %d\n", can01, can10);
  fflush(stdout);

  char* a=nullptr; char* b=nullptr;
  CK(hipSetDevice(0)); CK(hipMalloc(&a, bytes)); CK(hipMemset(a, 0xAB, bytes));
  CK(hipSetDevice(1)); CK(hipMalloc(&b, bytes)); CK(hipMemset(b, 0x00, bytes));
  CK(hipDeviceSynchronize());

  hipError_t pe01 = hipSetDevice(0) , pe0 = hipSuccess;
  pe0 = hipDeviceEnablePeerAccess(1, 0);
  hipError_t pe1 = hipDeviceEnablePeerAccess(0, 0);   // from device 1
  printf("enablePeerAccess 0->1: %s\n", hipGetErrorString(pe0));
  printf("enablePeerAccess 1->0: %s\n", hipGetErrorString(pe1));
  fflush(stdout);

  // memcpyPeer 0->1 (guard: report error, do not crash)
  std::vector<double> t01;
  CK(hipSetDevice(0));
  for (int r=0;r<reps;r++){
    hipEvent_t ea,eb; CK(hipEventCreate(&ea)); CK(hipEventCreate(&eb));
    CK(hipEventRecord(ea));
    hipError_t e = hipMemcpyPeer(b, 1, a, 0, bytes);
    if (e != hipSuccess){ printf("memcpyPeer 0->1 FAILED: %s\n", hipGetErrorString(e)); fflush(stdout); return 0; }
    CK(hipEventRecord(eb)); CK(hipEventSynchronize(eb));
    float ms=0; CK(hipEventElapsedTime(&ms,ea,eb));
    t01.push_back((double)bytes/(ms*1e-3)/1e9);
    CK(hipEventDestroy(ea)); CK(hipEventDestroy(eb));
  }
  printf("memcpyPeer 0->1: %.2f GB/s median\n", med(t01));
  fflush(stdout);
  char* h=nullptr; CK(hipHostMalloc(&h, bytes));
  CK(hipSetDevice(1)); CK(hipMemcpy(h, b, bytes, hipMemcpyDeviceToHost));
  int good = (h[<bus>]==(char)0xAB && h[bytes-1]==(char)0xAB);
  printf("content arrived: %s\n", good?"yes":"NO");
  CK(hipSetDevice(0)); CK(hipFree(a));
  CK(hipSetDevice(1)); CK(hipFree(b));
  CK(hipHostFree(h));
  printf("m0a done\n");
  return 0;
}
