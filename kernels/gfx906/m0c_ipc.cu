// m0c_ipc.cu — the FORK's exact cross-rank path, 2 devices same process:
// malloc staging on dev1, hipIpcGetMemHandle, open on dev0 with
// hipIpcMemLazyEnablePeerAccess (rdna_allreduce.cu lines 105-127), then a
// kernel on dev0 writing into the opened (dev1) staging. Reports every step.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CK(x) do { hipError_t e = (x); if (e != hipSuccess) { \
  fprintf(stderr, "FAIL %s:%d  %s -> %s\n", __FILE__, __LINE__, #x, \
          hipGetErrorString(e)); exit(1); } } while (0)

__global__ void poke(int* dst, int v, int n){
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) dst[i] = v;
}

int main(){
  int nd=0; CK(hipGetDeviceCount(&nd));
  printf("m0c: %d devices\n", nd);
  if (nd < 2) return 2;
  const int N = 1<<20;  // 4 MB int staging
  int* dev1 = nullptr;
  CK(hipSetDevice(1));
  CK(hipMalloc(&dev1, N*sizeof(int)));
  CK(hipMemset(dev1, 0, N*sizeof(int)));
  CK(hipDeviceSynchronize());
  hipIpcMemHandle_t handle;
  hipError_t ge = hipIpcGetMemHandle(&handle, dev1);
  printf("ipcGetMemHandle(dev1): %s\n", hipGetErrorString(ge));
  if (ge != hipSuccess) return 0;

  // open from device 0 (the fork passes hipIpcMemLazyEnablePeerAccess)
  int* opened = nullptr;
  CK(hipSetDevice(0));
  hipError_t oe = hipIpcOpenMemHandle((void**)&opened, handle,
                                      hipIpcMemLazyEnablePeerAccess);
  printf("ipcOpenMemHandle(dev0, LazyEnablePeerAccess): %s\n",
         hipGetErrorString(oe));
  if (oe != hipSuccess) {
    // try without the lazy peer flag to isolate peer-enable from IPC itself
    hipError_t oe2 = hipIpcOpenMemHandle((void**)&opened, handle, 0);
    printf("ipcOpenMemHandle(dev0, no flag): %s\n", hipGetErrorString(oe2));
    if (oe2 != hipSuccess) return 0;
  }

  // kernel on dev0 writes into the staging that lives on dev1
  CK(hipSetDevice(0));
  poke<<<dim3(64), dim3(256), 0, 0>>>(opened, 0x5A5A5A5A, N);
  hipError_t le = hipGetLastError();
  printf("kernel launch on dev0 into dev1 staging: %s\n",
         hipGetErrorString(le));
  if (le != hipSuccess) return 0;
  CK(hipDeviceSynchronize());
  // verify from dev1
  int* h = new int[<bus>];
  CK(hipSetDevice(1));
  CK(hipMemcpy(h, dev1, 4*sizeof(int), hipMemcpyDeviceToHost));
  CK(hipDeviceSynchronize());
  printf("staging words after dev0 write: %08x %08x %08x %08x (want 5a5a5a5a)\n",
         (unsigned)h[<bus>], (unsigned)h[<bus>], (unsigned)h[<bus>], (unsigned)h[<bus>]);
  delete[] h;
  CK(hipSetDevice(1)); CK(hipFree(dev1));
  printf("m0c done\n");
  return 0;
}
