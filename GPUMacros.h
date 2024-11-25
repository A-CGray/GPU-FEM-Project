#pragma once

#ifdef __CUDACC__
#define __SHARED__ __shared__
#define SYNC_THREADS __syncthreads();
#define __HOST_AND_DEVICE__ __host__ __device__
#define __DEVICE__ __device__
#define __GLOBAL__ __global__

#else
#define __SHARED__
#define SYNC_THREADS
#define __HOST_AND_DEVICE__
#define __DEVICE__
#define __GLOBAL__

#endif

#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
// Usage: put gpuErrchk(...) around cuda function calls
#define gpuErrchk(ans)                                                                                                 \
  {                                                                                                                    \
    gpuAssert((ans), __FILE__, __LINE__);                                                                              \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort)
      exit(code);
  }
}

void cuda_show_kernel_error() {
  auto err = cudaGetLastError();
  std::cout << "error code: " << err << "\n";
  std::cout << "error string: " << cudaGetErrorString(err) << "\n";
}
#endif
