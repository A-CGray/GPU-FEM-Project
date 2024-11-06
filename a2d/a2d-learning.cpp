#include "a2dcore.h"
#include <cstdlib>
#include <math.h>
#include <typeinfo>

using namespace A2D;

#define T double
#define N 3

T randomNum() { return T(std::rand()) / RAND_MAX; }

int main(int argc, char *argv[]) {
  // Start with a simple example, compute the first and second derivatives of sin(x) using forward and reverse AD
  ADObj<T> x(randomNum() * 2 * M_PI); // x needs to be an ADObj because we want to compute a derivative w.r.t it
  T refDeriv = cos(x.value());

  // To compute the derivative we need to make a stack?
  ADObj<T> sinx;
  auto sinxStack = MakeStack(Eval(sin(x), sinx));

  // To compute a derivative with reverse AD, set the seed in the output and run reverse, derivative will be in the seed
  // of the input
  sinx.bvalue() = 1.0;
  // Do the reverse AD, after this, d/dx(sin(x)) will be in x.bvalue()
  sinxStack.reverse();

  printf("sin(%f) = %f\n", x.value(), sinx.value());
  printf("d/dx(sin(x=%f)): RAD = %f, Reference = %f\n", x.value(), x.bvalue(), refDeriv);

  // Now let's try forward AD, set the seed in the input, run forward, then the derivative will be in the output
  x.bvalue() = 1.0;
  sinxStack.forward();
  printf("d/dx(sin(x=%f)): FAD = %f, Reference = %f\n", x.value(), sinx.bvalue(), refDeriv);

  return 0;
}
