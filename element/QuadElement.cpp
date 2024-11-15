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

    /**
     * @brief Given values at nodes, interpolate the values a point in the element
     *
     * @tparam valsPerNode The number of values defined at each node
     * @param paramCoord Parametric coordinate to interpolate to
     * @param nodalValues Values at the nodes (states, coordinates etc)
     * @param interpValues Values at the point in the element
     */
    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void interpolate(const numType paramCoord[numDim],
                                                      const numType nodalValues[valsPerNode * numNodes],
                                                      numType interpValues[valsPerNode]) {
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

    /**
     * @brief Given values at nodes, interpolate the gradient at a point in the element
     *
     * @tparam valsPerNode The number of values defined at each node
     * @param paramCoord Parametric coordinate to interpolate to
     * @param nodalValues Values at the nodes (states, coordinates etc)
     * @param interpGradValues Gradient at the point in the element (valsPerNode x 2)
     */
    template <int valsPerNode>
    __HOST_AND_DEVICE__ static const void interpolateGradient(const numType paramCoord[numDim],
                                                              const numType nodalValues[valsPerNode * numNodes],
                                                              numType interpGradValues[valsPerNode * numDim]) {
      for (int ii = 0; ii < valsPerNode * numDim; ii++) {
        interpGradValues[ii] = 0.0;
      }
      for (int jj = 0; jj < numNodes; jj++) {
        numType dNdx[2];
        basis.evalDeriv(paramCoord, jj, dNdx);
        for (int ii = 0; ii < valsPerNode; ii++) {
          interpGradValues[ii * numDim] += dNdx[0] * nodalValues[jj * valsPerNode + ii];
          interpGradValues[ii * numDim + 1] += dNdx[1] * nodalValues[jj * valsPerNode + ii];
        }
      }
    }

    /**
     * @brief Transform a gradient w.r.t parametric coordinates to a gradient w.r.t real coordinates
     *
     * @tparam numVals
     * @param paramCoord Parametric coordinate to transform at
     * @param nodeCoords Node coordinates
     * @param dfdxi Gradient w.r.t parametric coordinates
     * @param dfdx Gradient w.r.t real coordinates
     */
    template <int numVals>
    __HOST_AND_DEVICE__ static const void transformGradToRealSpace(const numType paramCoord[numDim],
                                                                   const numType nodeCoords[numNodes * numDim],
                                                                   const numType dfdxi[numVals * numDim],
                                                                   numType dfdx[numVals * numDim]) {}

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
    interpolateGradientToQuadPt(const int quadPtInd, const numType nodalValues[], numType interpGradValues[]) {}

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
