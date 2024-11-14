#include "../GPUMacros.h"
#include "adscalar.h"

#pragma once

/**
 * @brief Compute the value of the nth Lagrange polynomial at a given point x
 *
 * @tparam numType input and output numeric type
 * @tparam order order of the polynomial
 * @param x Parametric coordinate (-1 <= x <= 1)
 * @param nodeInd Index of the node/knot to compute the polynomial for
 * @return numType Value of the Lagrange polynomial at x
 */
template <typename numType, int order>
__HOST_AND_DEVICE__ numType lagrangePoly1d(const numType x, const int nodeInd) {
  numType result = 1.0;
  const numType xj = -1 + 2.0 * nodeInd / order;
  for (int ii = 0; ii < order + 1; ii++) {
    const numType xi = -1 + 2.0 * ii / order;
    if (ii != nodeInd) {
      result *= (x - xi) / (xj - xi);
    }
  }
  return result;
}

/**
 * @brief Compute the value and derivative of the nth Lagrange polynomial at a given point x
 *
 * @tparam numType input and output numeric type
 * @tparam order order of the polynomial
 * @param x Parametric coordinate (-1 <= x <= 1)
 * @param nodeInd Index of the node/knot to compute the polynomial for
 * @param N Variable to store the value of the polynomial in
 * @param dNdx Variable to store the derivative of the polynomial in
 */
template <typename numType, int order>
__HOST_AND_DEVICE__ void lagrangePoly1dDeriv(const numType x, const int nodeInd, numType &N, numType &dNdx) {
  A2D::ADScalar<numType, 1> input(x);
  input.deriv[0] = 1.0;
  A2D::ADScalar<numType, 1> result = lagrangePoly1d<A2D::ADScalar<numType, 1>, order>(input, nodeInd);
  N = result.value;
  dNdx = result.deriv[0];
}

/**
 * @brief Evaluate a 2D Lagrange polynomial at a given point
 *
 * @tparam numType input and output numeric type
 * @tparam order order of the polynomial
 * @param x coordinate in 2D parametric space (-1 <= x <= 1)
 * @param nodeXInd X index of the node to evaluate the polynomial for
 * @param nodeYInd Y index of the node to evaluate the polynomial for
 * @return numType Value of polynomial at x
 */
template <typename numType, int order>
__HOST_AND_DEVICE__ numType lagrangePoly2d(const numType x[2], const int nodeXInd, const int nodeYInd) {
  return lagrangePoly1d<numType, order>(x[0], nodeXInd) * lagrangePoly1d<numType, order>(x[1], nodeYInd);
}

/**
 * @brief Compute the value and derivative of a 2D Lagrange polynomial at a given point
 *
 * @tparam numType input and output numeric type
 * @tparam order order of the polynomial
 * @param x coordinate in 2D parametric space (-1 <= x <= 1)
 * @param nodeXInd X index of the node to evaluate the polynomial for
 * @param nodeYInd Y index of the node to evaluate the polynomial for
 * @param N Variable to store the value of the polynomial in
 * @param deriv Array to store the derivatives of the polynomial in
 */
template <typename numType, int order>
__HOST_AND_DEVICE__ void
lagrangePoly2dDeriv(const numType x[2], const int nodeXInd, const int nodeYInd, numType &N, numType deriv[2]) {
  numType Nx, Ny, dNdx, dNdy;
  lagrangePoly1dDeriv<numType, order>(x[0], nodeXInd, Nx, dNdx);
  lagrangePoly1dDeriv<numType, order>(x[1], nodeYInd, Ny, dNdy);
  deriv[0] = Ny * dNdx;
  deriv[1] = Nx * dNdy;
}

template <typename numType, int order>
class Lagrange2DBasis {
  public:
    static const numType eval(const numType x[2], const int nodeXInd, const int nodeYInd) {
      return lagrangePoly2d<numType, order>(x, nodeXInd, nodeYInd);
    }

    static const void evalDeriv(const numType x[2], const int nodeXInd, const int nodeYInd, numType deriv[2]) {
      lagrangePoly2dDeriv<numType, order>(x, nodeXInd, nodeYInd, deriv);
    }
};

// int main() {
//   const int order = 5;
//   const double x = 0.1234;

//   printf("Evaluating Lagrange shape functions of order %d at x = %f\n", order, x);
//   for (int ii = 0; ii < order + 1; ii++) {
//     double N, N2, dNdx;
//     N = lagrangePoly1d<double, order>(x, ii);
//     lagrangePoly1dDeriv<double, order>(x, ii, N2, dNdx);
//     printf("N_%d = % f, dN_%d/dx = % f\n", ii, N2, ii, dNdx);
//   }
//   return 0;
// }
