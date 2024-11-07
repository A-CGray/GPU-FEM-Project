#include "a2dcore.h"
#include "adscalar.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <typeinfo>

using namespace A2D;

#define T double
#define N 3

T randomNum() { return T(std::rand()) / RAND_MAX; }

template <typename I, typename numType>
void print_row_major_matrix(const std::string name, I height, I width, numType mat[]) {
  std::cout << name << ":\n";
  for (I i = 0; i < height; i++) {
    std::cout << "[ ";
    for (I j = 0; j < width; j++) {
      std::cout << std::setw(15) << mat[i * width + j];
    }
    std::cout << " ]\n";
  }
}

int main(int argc, char *argv[]) {
  // ==============================================================================
  // Scalar example: sin(x)
  // ==============================================================================
  // Start with a simple example, compute the first and second derivatives of sin(x) using forward and reverse AD
  A2DObj<T> x(randomNum() * 2 *
              M_PI); // x needs to be an A2DObj because we want to compute a second derivative w.r.t it
  T refDeriv = cos(x.value());
  T refDeriv2 = -sin(x.value());

  // To compute the derivative we need to make a stack?
  A2DObj<T> sinx;
  auto sinxStack = MakeStack(Eval(sin(x), sinx));

  // To compute a derivative with reverse AD, set the seed in the output and run reverse, derivative will be in the seed
  // of the input
  sinx.bvalue() = 1.0;
  // Do the reverse AD, after this, d/dx(sin(x)) will be in x.bvalue()
  sinxStack.reverse();

  printf("sin(%f) = %f\n", x.value(), sinx.value());
  printf("d/dx(sin(x=%f)): RAD = %f, Reference = %f\n", x.value(), x.bvalue(), refDeriv);

  // Now let's try forward AD, set the seed in the input, run forward, then the derivative will be in the output
  x.pvalue() = 1.0;
  sinxStack.hforward();
  printf("d/dx(sin(x=%f)): FAD = %f, Reference = %f\n", x.value(), sinx.pvalue(), refDeriv);

  // We can also do forward AD using the ADScalar type without making a stack
  ADScalar<T, 1> xScalar(x.value()), y;
  xScalar.deriv[0] = 1.0;
  y = sin(xScalar);
  printf("sin(%f) = %f\n", xScalar.value, y.value);
  printf("d/dx(sin(x=%f)): FAD-Scalar = %f, Reference = %f\n", xScalar.value, y.deriv[0], refDeriv);

  // --- second derivatives ---
  // Can compute second derivative of the stack using a hessian product
  sinxStack.bzero();
  sinxStack.hzero();
  x.pvalue() = 1.0;
  sinx.bvalue() = 1.0;
  sinxStack.hproduct();
  printf("d^2/dx^2(sin(x=%f)): hproduct = %f, Reference = %f\n", x.value(), x.hvalue(), refDeriv2);

  // ==============================================================================
  // Quadratic form: f(x) = 1/2 * x^T A x + b^T x
  // ==============================================================================
  // In order to make this a stack we need to make variables for every intermediate computation:
  // Ax = A*x
  // inner = dot(x, Ax)
  // bx = dot(b, x)
  // f = 0.5 * inner + bx
  Mat<T, N, N> A; // We could make this a SymMat, but MatVecMult isn't implemented for it yet
  Vec<T, N> b;
  A2DObj<Vec<T, N>> xVec, Ax;
  A2DObj<T> f, inner, bx;
  for (int ii = 0; ii < N; ii++) {
    b[ii] = randomNum();
    xVec.value()[ii] = randomNum();
    for (int jj = 0; jj <= ii; jj++) {
      A(ii, jj) = randomNum();
      A(jj, ii) = A(ii, jj);
    }
  }

  auto quadFormStack =
      MakeStack(MatVecMult(A, xVec, Ax), VecDot(xVec, Ax, inner), VecDot(b, xVec, bx), Eval(0.5 * inner + bx, f));

  // --- verify that df/dx = Ax+b ---
  T Axb[N];
  for (int ii = 0; ii < N; ii++) {
    Axb[ii] = b[ii];
    for (int jj = 0; jj < N; jj++) {
      Axb[ii] += A(ii, jj) * xVec.value()[jj];
    }
  }
  f.bvalue() = 1.0; // seed for reverse AD
  quadFormStack.reverse();

  // Print the results
  std::cout << "\n\nf(x) = 1/2 * x^T A x + b^T x, df/dx = [ ";
  for (int ii = 0; ii < N; ii++) {
    std::cout << xVec.bvalue()[ii] << " ";
  }
  std::cout << "], Reference = [ ";
  for (int ii = 0; ii < N; ii++) {
    std::cout << Axb[ii] << " ";
  }
  std::cout << "]" << std::endl;

  // --- Verify that d^2f/dx^2 = A ---
  Mat<T, N, N> Hessian;
  quadFormStack.bzero();
  f.bvalue() = 1.0;
  quadFormStack.hextract(xVec.pvalue(), xVec.hvalue(), Hessian);

  // Print the results
  print_row_major_matrix("d^2f/dx^2", N, N, Hessian.get_data());
  print_row_major_matrix("A", N, N, A.get_data());

  return 0;
}
