/*
=============================================================================
Benchmarking Lagrange polynomial evaluation
=============================================================================
@File    :   LagrangePolyBenchmark.cpp
@Date    :   2024/11/14
@Author  :   Alasdair Christison Gray
@Description : Comparing the fully generic templated implementation with specialised implementations for lower orders
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "LagrangeShapeFuncs.h"
#include <benchmark/benchmark.h>

double lagrangePoly1dOrder1(const double x, const int nodeInd) {
  switch (nodeInd) {
    case 0:
      return 0.5 * (1 - x);
    case 1:
      return 0.5 * (1 + x);
    default:
      return 0.0;
  }
}

void lagrangePoly1dOrder1Deriv(const double x, const int nodeInd, double &N, double &dNdx) {
  switch (nodeInd) {
    case 0:
      N = 0.5 * (1 - x);
      dNdx = -0.5;
    case 1:
      N = 0.5 * (1 + x);
      dNdx = 0.5;
    default:
      N = 0.0;
      dNdx = 0.0;
  }
}

double lagrangePoly1dOrder2(const double x, const int nodeInd) {
  switch (nodeInd) {
    case 0:
      return -0.5 * x * (1.0 - x);
    case 1:
      return (1.0 - x) * (1.0 + x);
    case 2:
      return 0.5 * (1.0 + x) * x;
    default:
      return 0.0;
  }
}

void lagrangePoly1dOrder2Deriv(const double x, const int nodeInd, double &N, double &dNdx) {
  switch (nodeInd) {
    case 0:
      N = -0.5 * x * (1.0 - x);
      dNdx = -0.5 + x;
    case 1:
      N = (1.0 - x) * (1.0 + x);
      dNdx = -2 * x;
    case 2:
      N = 0.5 * (1.0 + x) * x;
      dNdx = 0.5 + x;
    default:
      N = 0.0;
      dNdx = 0.0;
  }
}

double lagrangePoly1dOrder3(const double x, const int nodeInd) {
  switch (nodeInd) {
    case 0:
      return -(2.0 / 3.0) * (0.5 + x) * (0.5 - x) * (1.0 - x);
    case 1:
      return (4.0 / 3.0) * (1.0 + x) * (0.5 - x) * (1.0 - x);
    case 2:
      return (4.0 / 3.0) * (1.0 + x) * (0.5 + x) * (1.0 - x);
    case 3:
      return -(2.0 / 3.0) * (1.0 + x) * (0.5 + x) * (0.5 - x);
    default:
      return 0.0;
  }
}

void lagrangePoly1dOrder3Deriv(const double x, const int nodeInd, double &N, double &dNdx) {
  switch (nodeInd) {
    case 0:
      N = -(2.0 / 3.0) * (0.5 + x) * (0.5 - x) * (1.0 - x);
      dNdx = -2.0 * x * x + (4.0 / 3.0) * x + 1.0 / 6.0;
    case 1:
      N = (4.0 / 3.0) * (1.0 + x) * (0.5 - x) * (1.0 - x);
      dNdx = 4.0 * x * x - (4.0 / 3.0) * x - 4.0 / 3.0;
    case 2:
      N = (4.0 / 3.0) * (1.0 + x) * (0.5 + x) * (1.0 - x);
      dNdx = -4.0 * x * x - (4.0 / 3.0) * x + 4.0 / 3.0;
    case 3:
      N = -(2.0 / 3.0) * (1.0 + x) * (0.5 + x) * (0.5 - x);
      dNdx = 2.0 * x * x + (4.0 / 3.0) * x - 1.0 / 6.0;
    default:
      N = 0.0;
      dNdx = 0.0;
  }
}

double fRand(double fMin, double fMax) {
  double f = (double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

// Template function for benchmarking the evaluation of one of the lagrange polynomial functions
template <typename Func>
static void lagrangePolyBenchmarkTemplate(benchmark::State &state, Func func, const int order) {
  for (auto _ : state) {
    double result = 0.0;
    const double x = fRand(-1.0, 1.0);
    for (int ii = 0; ii < order + 1; ii++) {
      result += func(x, ii);
    }
    benchmark::DoNotOptimize(result);
  }
}

// Template function for benchmarking the evaluation of one of the lagrange polynomial functions
template <typename Func>
static void lagrangePolyDerivBenchmarkTemplate(benchmark::State &state, Func func, const int order) {
  for (auto _ : state) {
    double result = 0.0;
    double deriv = 0.0;
    double tmp1, tmp2;
    const double x = fRand(-1.0, 1.0);
    for (int ii = 0; ii < order + 1; ii++) {
      func(x, ii, tmp1, tmp2);
      result += tmp1;
      deriv += tmp2;
    }
    benchmark::DoNotOptimize(result);
    benchmark::DoNotOptimize(deriv);
  }
}

// Register benchmarks for each order
// --- First order ---
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Specialised First Order", lagrangePoly1dOrder1, 1);
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Generic First Order", lagrangePoly1d<double, 1>, 1);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Specialised First Order Deriv", lagrangePoly1dOrder1Deriv, 1);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Generic First Order Deriv", lagrangePoly1dDeriv<double, 1>, 1);

// --- Second order ---
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Specialised Second Order", lagrangePoly1dOrder2, 2);
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Generic Second Order", lagrangePoly1d<double, 2>, 2);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Specialised Second Order Deriv", lagrangePoly1dOrder2Deriv, 2);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Generic Second Order Deriv", lagrangePoly1dDeriv<double, 2>, 2);

// --- Third order ---
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Specialised Third Order", lagrangePoly1dOrder3, 3);
BENCHMARK_CAPTURE(lagrangePolyBenchmarkTemplate, "Generic Third Order", lagrangePoly1d<double, 3>, 3);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Specialised Third Order Deriv", lagrangePoly1dOrder3Deriv, 3);
BENCHMARK_CAPTURE(lagrangePolyDerivBenchmarkTemplate, "Generic Third Order Deriv", lagrangePoly1dDeriv<double, 3>, 3);

BENCHMARK_MAIN();
