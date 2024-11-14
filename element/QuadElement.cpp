/*
=============================================================================
Base Element class
=============================================================================
@File    :   Element.h
@Date    :   2024/11/12
@Author  :   Alasdair Christison Gray
@Description :
*/

// =============================================================================
// Standard Library Includes
// =============================================================================
#include <cstdio>

// =============================================================================
// Extension Includes
// =============================================================================
#include "../GPUMacros.h"
#include "GaussQuadrature.h"
#include "LagrangeShapeFuncs.h"

// =============================================================================
// Global constant definitions
// =============================================================================

// =============================================================================
// Function prototypes
// =============================================================================

template <typename numType, int order, int _numStates>
class QuadElement {
  public:
    static const int numNodes = (order + 1) * (order + 1);
    static const int numDim = 2;
    static const int numStates = _numStates;
    static const int numQuadPoints = (order + 1) * (order + 1);

    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void
    interpolate(const numType paramCoord[], const numType nodalValues[], numType interpValues[]) {
      for (int ii = 0; ii < valsPerNode; ii++) {
        interpValues[ii] = 0.0;
      }
      for (int jj = 0; jj < numNodes; jj++) {
        const numType N = basis.eval(paramCoord, jj);
        for (int ii = 0; ii < valsPerNode; ii++) {
          interpValues[ii] += N * nodalValues[jj * valsPerNode + ii];
        }
      }
    }

    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void
    interpolateToQuadPt(const int quadPtInd, const numType nodalValues[], numType interpValues[]) {
      for (int ii = 0; ii < valsPerNode; ii++) {
        interpValues[ii] = 0.0;
      }
      for (int jj = 0; jj < numNodes; jj++) {
        const numType N = getQuadPtShapeFunc(quadPtInd, jj);
        for (int ii = 0; ii < valsPerNode; ii++) {
          interpValues[ii] += N * nodalValues[jj * valsPerNode + ii];
        }
      }
    }

    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void
    interpolateGradient(const numType paramCoord[], const numType nodalValues[], numType interpGradValues[]) {}

    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void
    interpolateGradientToQuadPt(const int quadPtInd, const numType nodalValues[], numType interpGradValues[]) {}

    // ==============================================================================
    // Specialized functions for evaluation at quadrature points
    // ==============================================================================
    __HOST_AND_DEVICE__ static const void getQuadPtCoord(const int quadPtInd, numType quadPtCoord[]) {
      quadPtCoord[0] = quadrature.coords[quadPtInd % (order + 1)];
      quadPtCoord[1] = quadrature.coords[quadPtInd / (order + 1)];
    }

    __HOST_AND_DEVICE__ static const numType getQuadPtWeight(const int quadPtInd) {
      return quadrature.weights[quadPtInd % (order + 1)] * quadrature.weights[quadPtInd / (order + 1)];
    }

    __HOST_AND_DEVICE__ static const numType getQuadPtShapeFunc(const int quadPtInd, const int nodeInd) {
      const int nodeXInd = nodeInd % (order + 1);
      const int nodeYInd = nodeInd / (order + 1);
      numType xParam[2];
      getQuadPtCoord(quadPtInd, xParam);
      return basis.eval(xParam, nodeXInd, nodeYInd);
    }

    __HOST_AND_DEVICE__ static const void
    getQuadPtShapeFuncDeriv(const int quadPtInd, const int nodeInd, numType dNdx[2]) {
      const int nodeXInd = nodeInd % (order + 1);
      const int nodeYInd = nodeInd / (order + 1);
      numType xParam[2];
      getQuadPtCoord(quadPtInd, xParam);
      basis.evalDeriv(xParam, nodeXInd, nodeYInd, dNdx);
    }

  private:
    static constexpr GaussQuadrature quadrature = GaussQuadrature<order + 1>();
    static constexpr Lagrange2DBasis basis = Lagrange2DBasis<numType, order>();
};

int main() {
  const int ORDER = 3;
  QuadElement<double, ORDER, 2> quadElement;
  printf("I am a quad element with %d nodes, %d dimensions, %d states, and %d quadrature points\n",
         quadElement.numNodes,
         quadElement.numDim,
         quadElement.numStates,
         quadElement.numQuadPoints);
  printf("Quad points:\n");
  for (int ii = 0; ii < quadElement.numQuadPoints; ii++) {
    double quadPtCoord[2];
    quadElement.getQuadPtCoord(ii, quadPtCoord);
    printf("Coord = (% f, % f), weight = %f\n", quadPtCoord[0], quadPtCoord[1], quadElement.getQuadPtWeight(ii));
  }
  return 0;
}
