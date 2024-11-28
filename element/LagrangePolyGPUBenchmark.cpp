/*
=============================================================================

=============================================================================
@File    :   LagrangePolyGPUBenchmark.cpp
@Date    :   2024/11/28
@Author  :   Alasdair Christison Gray
@Description :
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "GPUMacros.h"
#include "LagrangeShapeFuncs.h"
#include <benchmark/benchmark.h>

// =============================================================================
// Global constant definitions
// =============================================================================
const int ARRAY_SIZE = 100000;
const int ARRAY_MEM_SIZE = ARRAY_SIZE * sizeof(double);
const int NUM_EVALS = 2000;
#define GLOBAL_THREAD_ID (blockIdx.x * blockDim.x + threadIdx.x)

// =============================================================================
// Function prototypes
// =============================================================================
template <int order>
__global__ void evalLagrangePoly2dDeriv(double *x, double *N, double *dNdxi) {
  const int arrayInd = GLOBAL_THREAD_ID;
  const int numKnots = order + 1;
  if (arrayInd < ARRAY_SIZE) {
    double localN = 0.;
    double localdNdxi[2] = {0.0, 0.0};
    const int nodeXInd = (arrayInd / numKnots) % numKnots;
    const int nodeYInd = arrayInd % numKnots;
    for (int ii = 0; ii < NUM_EVALS; ii++) {
      const int xPtInd = (arrayInd + ii) % ARRAY_SIZE;
      double nTemp, dNdxTemp[2];
      lagrangePoly2dDeriv<double, order>(&x[2 * xPtInd], nodeXInd, nodeYInd, nTemp, dNdxTemp);
      localN += nTemp;
      localdNdxi[0] += dNdxTemp[0];
      localdNdxi[1] += dNdxTemp[1];
    }

    N[arrayInd] += localN;
    dNdxi[arrayInd] = localdNdxi[0];
    dNdxi[arrayInd + 1] = localdNdxi[1];
  }
}

double fRand(double fMin, double fMax) {
  double f = (double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

// =============================================================================
// Main
// =============================================================================

template <int order, int blockSize>
void lagrangePolyGPUEvalBenchmarkTemplate(benchmark::State &state) {
  // Create CPU arrays
  double x[2 * ARRAY_SIZE], N[ARRAY_SIZE], dNdx[2 * ARRAY_SIZE];
  for (int ii = 0; ii < ARRAY_SIZE; ii++) {
    x[2 * ii] = fRand(-1, 1);
    x[2 * ii + 1] = fRand(-1, 1);
    N[ii] = 0.;
    dNdx[2 * ii] = 0;
    dNdx[2 * ii + 1] = 0;
  }

  // Create GPU arrays and copy data
  double *d_x;
  double *d_N;
  double *d_dNdx;
  cudaMalloc(&d_x, ARRAY_MEM_SIZE);
  cudaMemcpy(d_x, x, 2 * ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
  cudaMalloc(&d_N, ARRAY_MEM_SIZE);
  cudaMemcpy(d_N, N, ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);
  cudaMalloc(&d_dNdx, 2 * ARRAY_MEM_SIZE);
  cudaMemcpy(d_dNdx, dNdx, 2 * ARRAY_MEM_SIZE, cudaMemcpyHostToDevice);

  // Figure out the number of blocks to launch
  const int numBlocks = getNumBlocks(ARRAY_SIZE, blockSize);

  // Do a kernel launch to warm up the GPU
  evalLagrangePoly2dDeriv<order><<<numBlocks, blockSize>>>(d_x, d_N, d_dNdx);
  gpuErrchk(cudaDeviceSynchronize());

  // Evaluate the lagrange polynomials
  for (auto _ : state) {
    evalLagrangePoly2dDeriv<order><<<numBlocks, blockSize>>>(d_x, d_N, d_dNdx);
    cudaDeviceSynchronize();
  }

  // Copy back N to CPU, sum, and don't optimize
  cudaMemcpy(d_N, N, ARRAY_MEM_SIZE, cudaMemcpyDeviceToHost);
  cudaMemcpy(d_dNdx, dNdx, 2 * ARRAY_MEM_SIZE, cudaMemcpyDeviceToHost);
  double sum = 0;
  for (int ii = 0; ii < ARRAY_SIZE; ii++) {
    sum += N[ii];
  }
  benchmark::DoNotOptimize(sum);
  benchmark::DoNotOptimize(dNdx);
}

BENCHMARK(lagrangePolyGPUEvalBenchmarkTemplate<1, 256>)->Unit(benchmark::kMicrosecond)->Name("Order 1 blockSize 256");
BENCHMARK(lagrangePolyGPUEvalBenchmarkTemplate<2, 256>)->Unit(benchmark::kMicrosecond)->Name("Order 2 blockSize 256");
BENCHMARK(lagrangePolyGPUEvalBenchmarkTemplate<3, 256>)->Unit(benchmark::kMicrosecond)->Name("Order 3 blockSize 256");
BENCHMARK(lagrangePolyGPUEvalBenchmarkTemplate<4, 256>)->Unit(benchmark::kMicrosecond)->Name("Order 4 blockSize 256");

BENCHMARK_MAIN();
