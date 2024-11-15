#pragma once

#ifndef __CUDACC__
#define __SHARED__
#define SYNC_THREADS
#define __HOST_AND_DEVICE__
#define __DEVICE__
#define A2D_FUNCTION

#else
#define __SHARED__ __shared__
#define SYNC_THREADS __syncthreads();
#define __HOST_AND_DEVICE__ __host__ __device__
#define __DEVICE__ __device__
#define A2D_FUNCTION __host__ __device__

#endif
