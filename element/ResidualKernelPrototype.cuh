#include "a2dcore.h"

/**
 * @brief Assemble the residual for a plane stress element
 *
 * @tparam numNodes Number of nodes in each element
 * @tparam numStates Number of states at each node (Should always be 2?)
 * @tparam numQuadPts Number of quad points in each element
 * @tparam numDim Number of dimensions (Should always be 2?)
 * @param connectivity Node connectivity array
 * @param numElements Number of elements
 * @param states Global nodal state array
 * @param nodeCoords Global nodal coordinate array (assumed to be in numDim-D, not necessarily 3D)
 * @param quadPtWeights Weights for each quad point
 * @param quadPointdNdxi Basis function gradients at each quad point
 * @param E Elastic modulus
 * @param nu Poisson's ratio
 * @param t Thickness
 * @param residual Global residual array
 */
template <int numNodes,
          int numStates,
          int numQuadPts,
          int numDim,
          template <GreenStrainType etype = GreenStrainType::LINEAR>>
__global__ assemblePlaneStressResidual(const int *const connectivity,
                                       const int numElements,
                                       const double *const states,
                                       const double *const nodeCoords,
                                       const double *const quadPtWeights,
                                       const double *const quadPointdNdxi,
                                       const double E,
                                       const double nu,
                                       const double t,
                                       double *const residual);

/**
 * @brief Transform the sensitivity of a value w.r.t the state gradient to a sensitivity w.r.t the nodal states and add
 * it to an existing array
 *
 * @tparam numNodes
 * @tparam numVals
 * @tparam numDim
 * @param stateGradSens
 * @param Jinv
 * @param dNdxi
 * @param nodalValSens
 * @return __DEVICE__
 */
template <int numNodes, int numVals, int numDim>
__DEVICE__ void addTransformStateGradSens(const double stateGradSens[numVals * numDim],
                                          const double Jinv[numDim * numDim],
                                          const double dNdxi[numNodes * numDim],
                                          double nodalValSens[]);

// dudxi = localNodeValues^T * dNdxi
template <int numNodes, int numVals, int numDim>
__DEVICE__ void interpParamGradient(const double nodalValues[numNodes * numVals],
                                    const double dNdxi[numNodes * numDim],
                                    double dudxi[numVals * numDim]);

// dudx = localNodeValues^T * dNdxi * Jinv
template <int numNodes, int numVals, int numDim>
__DEVICE__ void interpRealGradient(const double nodalValues[numNodes * numVals],
                                   const double dNdxi[numNodes * numDim],
                                   const double Jinv[numDim * numDim],
                                   double dudx[numVals * numDim]);

template <GreenStrainType etype = GreenStrainType::LINEAR>
__DEVICE__ void planeStressWeakRes(const double uPrime[4],
                                   const double vPrime[4],
                                   const double E,
                                   const double nu,
                                   const double scale,
                                   double residual[4]);
