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
__HOST_AND_DEVICE__ inline numType lagrangePoly1d(const numType x, const int nodeInd) {
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

template <>
__HOST_AND_DEVICE__ inline double lagrangePoly1d<double, 1>(const double x, const int nodeInd) {
  // Precomputed coefficients for each basis function
  static const double coeffs[2][2] = {
      {-0.5, 0.5}, // N0
      {0.5, 0.5},  // N1
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  // Polynomial evaluation: a*x + b
  return fma(poly[0], x, poly[1]);
}

template <>
__HOST_AND_DEVICE__ inline double lagrangePoly1d<double, 2>(const double x, const int nodeInd) {
  // Precomputed coefficients for each basis function
  static const double coeffs[3][3] = {
      {0.5, -0.5, 0.0}, // N0
      {-1.0, 0.0, 1.0}, // N1
      {0.5, 0.5, 0.0},  // N2
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  // Polynomial evaluation: a*x^2 + b*x + c
  return fma(poly[0], x * x, fma(poly[1], x, poly[2]));
}

template <>
__HOST_AND_DEVICE__ inline double lagrangePoly1d<double, 3>(const double x, const int nodeInd) {
  // Precomputed coefficients for each basis function
  static const double coeffs[4][4] = {
      {-0.5625, 0.5625, 0.0625, -0.0625}, // N0
      {1.6875, -0.5625, -1.6875, 0.5625}, // N1
      {-1.6875, -0.5625, 1.6875, 0.5625}, // N2
      {0.5625, 0.5625, -0.0625, -0.0625}  // N3
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  const double x2 = x * x;
  const double x3 = x2 * x;

  // Polynomial evaluation: a*x^3 + b*x^2 + c*x + d
  return fma(poly[0], x3, fma(poly[1], x2, fma(poly[2], x, poly[3])));
}

template <>
__HOST_AND_DEVICE__ inline double lagrangePoly1d<double, 4>(const double x, const int nodeInd) {
  // Precomputed coefficients for each basis function
  static const double coeffs[5][5] = {
      {2.0 / 3.0, -2.0 / 3.0, -1.0 / 6.0, 1.0 / 6.0, 0.0}, // N0
      {-8.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, -4.0 / 3.0, 0.0}, // N1
      {4.0, 0.0, -5.0, 0.0, 1.0},                          // N2
      {-8.0 / 3.0, -4.0 / 3.0, 8.0 / 3.0, 4.0 / 3.0, 0.0}, // N3
      {2.0 / 3.0, 2.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0, 0.0}  // N4
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x2 * x2;

  // Polynomial evaluation: a*x^4 + b*x^3 + c*x^2 + d*x + e
  return fma(poly[0], x4, fma(poly[1], x3, fma(poly[2], x2, fma(poly[3], x, poly[4]))));
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
__HOST_AND_DEVICE__ inline void lagrangePoly1dDeriv(const numType x, const int nodeInd, numType &N, numType &dNdx) {
  A2D::ADScalar<numType, 1> input(x);
  input.deriv[0] = 1.0;
  A2D::ADScalar<numType, 1> result = lagrangePoly1d<A2D::ADScalar<numType, 1>, order>(input, nodeInd);
  N = result.value;
  dNdx = result.deriv[0];
}

template <>
__HOST_AND_DEVICE__ inline void
lagrangePoly1dDeriv<double, 1>(const double x, const int nodeInd, double &N, double &dNdx) {
  // Precomputed coefficients for each basis function
  static const double coeffs[2][2] = {
      {-0.5, 0.5}, // N0
      {0.5, 0.5},  // N1
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  // Polynomial evaluation: a*x + b
  N = fma(poly[0], x, poly[1]);

  // Derivative: a
  dNdx = poly[0];
}

template <>
__HOST_AND_DEVICE__ inline void
lagrangePoly1dDeriv<double, 2>(const double x, const int nodeInd, double &N, double &dNdx) {
  // Precomputed coefficients for each basis function
  static const double coeffs[3][3] = {
      {0.5, -0.5, 0.0}, // N0
      {-1.0, 0.0, 1.0}, // N1
      {0.5, 0.5, 0.0},  // N2
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  // Polynomial evaluation: a*x^2 + b*x + c
  N = fma(poly[0], x * x, fma(poly[1], x, poly[2]));

  // Derivative: 2*a*x + b
  dNdx = fma(2 * poly[0], x, poly[1]);
}

template <>
__HOST_AND_DEVICE__ inline void
lagrangePoly1dDeriv<double, 3>(const double x, const int nodeInd, double &N, double &dNdx) {
  // Precomputed coefficients for each basis function
  static const double coeffs[4][4] = {
      {-0.5625, 0.5625, 0.0625, -0.0625}, // N0
      {1.6875, -0.5625, -1.6875, 0.5625}, // N1
      {-1.6875, -0.5625, 1.6875, 0.5625}, // N2
      {0.5625, 0.5625, -0.0625, -0.0625}  // N3
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  const double x2 = x * x;
  const double x3 = x2 * x;

  // Polynomial evaluation: a*x^3 + b*x^2 + c*x + d
  N = fma(poly[0], x3, fma(poly[1], x2, fma(poly[2], x, poly[3])));

  // Derivative: 3*a*x^2 + 2*b*x + c
  dNdx = fma(3 * poly[0], x2, fma(2 * poly[1], x, poly[2]));
}

template <>
__HOST_AND_DEVICE__ inline void
lagrangePoly1dDeriv<double, 4>(const double x, const int nodeInd, double &N, double &dNdx) {
  // Precomputed coefficients for each basis function
  static const double coeffs[5][5] = {
      {2.0 / 3.0, -2.0 / 3.0, -1.0 / 6.0, 1.0 / 6.0, 0.0}, // N0
      {-8.0 / 3.0, 4.0 / 3.0, 8.0 / 3.0, -4.0 / 3.0, 0.0}, // N1
      {4.0, 0.0, -5.0, 0.0, 1.0},                          // N2
      {-8.0 / 3.0, -4.0 / 3.0, 8.0 / 3.0, 4.0 / 3.0, 0.0}, // N3
      {2.0 / 3.0, 2.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0, 0.0}  // N4
  };

  // Get coefficients for this node
  const double *poly = coeffs[nodeInd];

  const double x2 = x * x;
  const double x3 = x2 * x;
  const double x4 = x2 * x2;

  // Polynomial evaluation: a*x^4 + b*x^3 + c*x^2 + d*x + e
  N = fma(poly[0], x4, fma(poly[1], x3, fma(poly[2], x2, fma(poly[3], x, poly[4]))));

  // Derivative: 4*a*x^3 + 3*b^2 + 2*c*x + d
  dNdx = fma(4 * poly[0], x3, fma(3 * poly[1], x2, fma(2 * poly[2], x, poly[3])));
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
__HOST_AND_DEVICE__ inline numType lagrangePoly2d(const numType x[2], const int nodeXInd, const int nodeYInd) {
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
__HOST_AND_DEVICE__ inline void
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
//   const int order = 1;
//   const double x = 0.1234;

//   printf("Evaluating Lagrange shape functions of order %d at x = %f\n", order, x);
//   for (int ii = 0; ii < order + 1; ii++) {
//     double N, N2, dNdx;
//     N = lagrangePoly1d<double, order>(x, ii);
//     lagrangePoly1dDeriv<double, order>(x, ii, N2, dNdx);
//     printf("N_%d = % f, dN_%d/dx = % f\n", ii, N, ii, dNdx);
//   }
//   return 0;
// }
