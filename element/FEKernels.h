#pragma once

#include "GPUMacros.h"
#include "GaussQuadrature.h"
#include "LagrangeShapeFuncs.h"
#include "a2dcore.h"
#include "adscalar.h"

template <typename T>
__HOST_AND_DEVICE__ void printMat(const T mat[], const int numRows, const int numCols) {
  for (int ii = 0; ii < numRows; ii++) {
    printf("[");
    for (int jj = 0; jj < numCols; jj++) {
      printf(" % 11.7e, ", mat[ii * numCols + jj]);
    }
    printf("]\n");
  }
}

template <typename T, int N, int M>
__HOST_AND_DEVICE__ void printMat(const A2D::Mat<T, N, M> mat) {
  printMat(mat.get_data(), N, M);
}

template <int numNodes, int numStates, int numDim>
__DEVICE__ void gatherElementData(const int *const connPtr,
                                  const int *const conn,
                                  const int elementInd,
                                  const double *const states,
                                  const double *const nodeCoords,
                                  A2D::Mat<double, numNodes, numStates> &localStates,
                                  A2D::Mat<double, numNodes, numDim> &localCoords) {
  for (int ii = 0; ii < numNodes; ii++) {
    const int nodeInd = conn[connPtr[elementInd] + ii];
    for (int jj = 0; jj < numDim; jj++) {
      localCoords(ii, jj) = nodeCoords[nodeInd * numDim + jj];
    }
    for (int jj = 0; jj < numStates; jj++) {
      localStates(ii, jj) = states[nodeInd * numStates + jj];
    }
  }
}

template <int numNodes, int numStates>
__DEVICE__ void scatterResidual(const int *const connPtr,
                                const int *const conn,
                                const int elementInd,
                                A2D::Mat<double, numNodes, numStates> &localRes,
                                double *const residual) {
  for (int ii = 0; ii < numNodes; ii++) {
    const int nodeInd = conn[connPtr[elementInd] + ii];
    for (int jj = 0; jj < numStates; jj++) {
#ifndef __CUDACC__
#pragma omp atomic
      residual[nodeInd * numStates + jj] += localRes[ii * numStates + jj];
#else
      atomicAdd(&residual[nodeInd * numStates + jj], localRes[ii * numStates + jj]);
#endif
    }
  }
}

template <int numNodes, int numStates>
__DEVICE__ void scatterMat(const int elementInd,
                           const int *const map,
                           A2D::Mat<double, numNodes * numStates, numNodes * numStates> localMat,
                           double *const matEntries) {
#ifndef NDEBUG
  printf("\nScattering matrix for element %d\n", elementInd);
#endif
  const int numDOF = numNodes * numStates;
  for (int iNode = 0; iNode < numNodes; iNode++) {
    for (int jNode = 0; jNode < numNodes; jNode++) {
      // We use the map to get the index in the BCSR data array that the local matrix block coupling nodes i and j
      // should be added to. Each block is numStates x numStates
      double *const globalMatBlock = &matEntries[map[iNode * numNodes + jNode]];
#ifndef NDEBUG
      printf("Block for node i = %d, node j = %d starts at A[%d]\n", iNode, jNode, map[iNode * numNodes + jNode]);
#endif
      for (int ii = 0; ii < numStates; ii++) {
        const int localMatRow = iNode * numStates + ii;
        for (int jj = 0; jj < numStates; jj++) {
          const int localMatCol = jNode * numStates + jj;
          const int localMatInd = localMatRow * numDOF + localMatCol;
          const int globalMatBlockInd = ii * numStates + jj;
#ifndef NDEBUG
          printf("Adding localMat[%d, %d] = %f to globalMatBlock[%d, %d]\n",
                 localMatRow,
                 localMatCol,
                 localMat[localMatInd],
                 ii,
                 jj);
#endif

#ifdef __CUDACC__
          atomicAdd(&globalMatBlock[globalMatBlockInd], localMat[localMatInd]);
#else
          globalMatBlock[globalMatBlockInd] += localMat[localMatInd];
#endif
        }
      }
    }
  }
}

template <int order, int numVals, int numDim, bool atomic = false>
__DEVICE__ void addTransformStateGradSens(const double xi[numDim],
                                          const A2D::Mat<double, numVals, numDim> stateGradSens,
                                          const A2D::Mat<double, numDim, numDim> JInv,
                                          double nodalValSens[(order + 1) * (order + 1) * numVals]) {
  constexpr int numNodes = (order + 1) * (order + 1);

  // dfdq = NPrimeParam * J^-1 * dfduPrime^T

  // Number of ops to compute (NPrimeParam * J^-1) * dfduPrime^T
  constexpr int cost1 = numNodes * numDim * numDim + numDim * numDim * numVals;
  // Number of ops to compute NPrimeParam * (J^-1 * dfduPrime^T)
  constexpr int cost2 = numDim * numDim * numVals + numNodes * numDim * numVals;

  if constexpr (cost1 > cost2) {
    // --- Compute J^-1 * dfduPrime^T ---
    A2D::Mat<double, numDim, numVals> temp;
    A2D::MatMatMult<A2D::MatOp::NORMAL, A2D::MatOp::TRANSPOSE>(JInv, stateGradSens, temp);
    // for (int ii = 0; ii < numDim; ii++) {
    //   for (int jj = 0; jj < numVals; jj++) {
    //     temp[ii, jj] = 0.0;
    //     for (int kk = 0; kk < numDim; kk++) {
    //       temp[ii, jj] += JInv[ii * numDim + kk] * stateGradSens[jj * numVals + kk];
    //     }
    //   }
    // }
    // --- Add NPrimeParam * temp ---
    // A2D doesn't have the ability to do in-place addition of a MatMat product so we'll just do it manually
    for (int nodeYInd = 0; nodeYInd < (order + 1); nodeYInd++) {
      for (int nodeXInd = 0; nodeXInd < (order + 1); nodeXInd++) {
        const int nodeInd = nodeYInd * (order + 1) + nodeXInd;
        double dNidxi[numDim], N;
        lagrangePoly2dDeriv<double, order>(xi, nodeXInd, nodeYInd, N, dNidxi);

        for (int jj = 0; jj < numVals; jj++) {
          double sum = 0.0;
          for (int kk = 0; kk < numDim; kk++) {
            sum += dNidxi[kk] * temp(kk, jj);
          }
          if constexpr (atomic) {
#ifdef __CUDACC__
            atomicAdd(&nodalValSens[nodeInd * numVals + jj], sum);
#else
#pragma omp atomic
            nodalValSens[nodeInd * numVals + jj] += sum;
#endif
          }
          else {
            nodalValSens[nodeInd * numVals + jj] += sum;
          }
        }
      }
    }
  }
  else {
    // --- Compute (NPrimeParam * J^-1) ---
    A2D::Mat<double, numNodes, numDim> temp;
    for (int nodeYInd = 0; nodeYInd < (order + 1); nodeYInd++) {
      for (int nodeXInd = 0; nodeXInd < (order + 1); nodeXInd++) {
        const int nodeInd = nodeYInd * (order + 1) + nodeXInd;
        double dNidxi[numDim], N;
        lagrangePoly2dDeriv<double, order>(xi, nodeXInd, nodeYInd, N, dNidxi);
        for (int jj = 0; jj < numDim; jj++) {
          for (int kk = 0; kk < numDim; kk++) {
            temp(nodeInd, jj) += dNidxi[kk] * JInv(kk, jj);
          }
        }
      }
    }
    // --- Add temp * dfduPrime^T ---
    // A2D doesn't have the ability to do in-place addition of a MatMat product so we'll just do it manually
    for (int ii = 0; ii < numNodes; ii++) {
      for (int jj = 0; jj < numVals; jj++) {
        double sum = 0.0;
        for (int kk = 0; kk < numDim; kk++) {
          sum += temp(ii, kk) * stateGradSens(jj, kk);
        }
        if constexpr (atomic) {
#ifdef __CUDACC__
          atomicAdd(&nodalValSens[ii * numVals + jj], sum);
#else
#pragma omp atomic
          nodalValSens[ii * numVals + jj] += sum;
#endif
        }
        else {
          nodalValSens[ii * numVals + jj] += sum;
        }
      }
    }
  }
}

// Ass above but only computes the sensitivity w.r.t a single nodal state
template <int order, int numVals, int numDim, bool atomic = false>
__DEVICE__ void addTransformStateGradSens(const double xi[numDim],
                                          const A2D::Mat<double, numVals, numDim> stateGradSens,
                                          const A2D::Mat<double, numDim, numDim> JInv,
                                          const int nodeXInd,
                                          const int nodeYInd,
                                          const int stateInd,
                                          double &nodalValSens) {
  double dNidxi[numDim], N;
  lagrangePoly2dDeriv<double, order>(xi, nodeXInd, nodeYInd, N, dNidxi);
  double sum = 0;
  for (int ii = 0; ii < numDim; ii++) {
    for (int jj = 0; jj < numDim; jj++) {
      sum += dNidxi[ii] * JInv(ii, jj) * stateGradSens(stateInd, jj);
    }
  }
  if constexpr (atomic) {
#ifdef __CUDACC__
    atomicAdd(&nodalValSens, sum);
#else
#pragma omp atomic
    nodalValSens += sum;
#endif
  }
  else {
    nodalValSens += sum;
  }
}

// dudxi = localNodeValues^T * dNdxi
template <int order, int numVals, int numDim>
__DEVICE__ void interpParamGradient(const double xi[numDim],
                                    const double nodalValues[(order + 1) * (order + 1) * numVals],
                                    A2D::Mat<double, numVals, numDim> &dudxi) {
  dudxi.zero();
  for (int nodeYInd = 0; nodeYInd < (order + 1); nodeYInd++) {
    for (int nodeXInd = 0; nodeXInd < (order + 1); nodeXInd++) {
      const int nodeInd = nodeYInd * (order + 1) + nodeXInd;
      double dNidxi[numDim], N;
      lagrangePoly2dDeriv<double, order>(xi, nodeXInd, nodeYInd, N, dNidxi);
      for (int ii = 0; ii < numVals; ii++) {
        for (int jj = 0; jj < numDim; jj++) {
          dudxi[ii * numDim + jj] += nodalValues[nodeInd * numVals + ii] * dNidxi[jj];
        }
      }
    }
  }

  // for (int ii = 0; ii < numVals; ii++) {
  //   for (int jj = 0; jj < numDim; jj++) {
  //     dudxi[ii, jj] = 0.;
  //     for (int kk = 0; kk < numNodes; kk++) {
  //       dudxi[ii * numDim + jj] += nodalValues[kk * numVals + ii] * dNdxi[kk * numDim + jj]
  //     }
  //   }
  // }
  // A2D::MatMatMult<A2D::MatOp::TRANSPOSE, A2D::MatOp::NORMAL>(nodalValues, dNdxi, dudxi);
}

// dudx = localNodeValues^T * dNdxi * JInv
template <int order, int numVals, int numDim>
__DEVICE__ void interpRealGradient(const double xi[numDim],
                                   const double nodalValues[(order + 1) * (order + 1) * numVals],
                                   const A2D::Mat<double, numDim, numDim> JInv,
                                   A2D::Mat<double, numVals, numDim> &dudx) {
  // dudxi = localNodeStates^T * dNdxi
  // dudx = dudxi * J^-1 = localNodeStates^T * dNdxi * J^-1
  // most efficient (in terms of FLOPS) to compute dudxi = localNodeStates^T * dNdxi first, then dudx = dudxi *
  // J^-1

  // dudxi = localNodeValues ^ T * dNdxi
  A2D::Mat<double, numVals, numDim> dudxi;
  interpParamGradient<order, numVals, numDim>(xi, nodalValues, dudxi);

  // dudx = dudxi * J^-1
  // for (int ii = 0; ii < numVals; ii++) {
  //   for (int jj = 0; jj < numDim) {
  //     dudx[ii * numDim + jj] = 0.0;
  //     for (int kk = 0; kk < numDim; kk++) {
  //       dudx[ii * numDim + jj] += dudxi[ii * numDim + kk] * JInv[kk * numDim + jj];
  //     }
  //   }
  // }
  A2D::MatMatMult<A2D::MatOp::NORMAL, A2D::MatOp::NORMAL>(dudxi, JInv, dudx);
}

template <A2D::GreenStrainType strainType, typename numType>
__DEVICE__ void planeStressWeakRes(const A2D::Mat<numType, 2, 2> uPrime,
                                   const double E,
                                   const double nu,
                                   const double t,
                                   const double scale,
                                   A2D::Mat<numType, 2, 2> &residual) {
  A2D::ADObj<A2D::Mat<numType, 2, 2>> uPrimeMat(uPrime);
  A2D::ADObj<A2D::SymMat<numType, 2>> strain, stress;
  A2D::ADObj<numType> energy;

  // NOTE: For 3D elasticity, the stress is: sigma = 2 * mu * epsilon + lambda * tr(epsilon) * I, however, for plane
  // stress we have to use a slightly different form to account for the assumed strain in the out of plane strain
  // component: sigma = 2 * mu * epsilon + (2*mu*nu/(1-nu)) * tr(epsilon) * I
  const double mu = 0.5 * E / (1.0 + nu);
  const double lambda = 2 * mu * nu / (1.0 - nu);

  auto stack = A2D::MakeStack(A2D::MatGreenStrain<strainType>(uPrimeMat, strain),
                              SymIsotropic(mu, lambda, strain, stress),
                              SymMatMultTrace(strain, stress, energy));

  // Compute strain energy derivative w.r.t state gradient
  energy.bvalue() = 0.5 * scale * t; // Set the seed value (0.5 because energy = 0.5 * sigma : epsilon)

  stack.reverse(); // Reverse mode AD through the stack

  for (int ii = 0; ii < 4; ii++) {
    residual[ii] = uPrimeMat.bvalue()[ii];
  }
}

template <int order, int numStates, int numDim, A2D::GreenStrainType strainType = A2D::GreenStrainType::LINEAR>
void assemblePlaneStressResidual(const int *const connPtr,
                                 const int *const conn,
                                 const int numElements,
                                 const double *const states,
                                 const double *const nodeCoords,
                                 const double E,
                                 const double nu,
                                 const double t,
                                 double *const residual) {
  constexpr int numNodes = (order + 1) * (order + 1);
#pragma omp parallel for
  for (int elementInd = 0; elementInd < numElements; elementInd++) {
    // Get the element nodal states and coordinates
    A2D::Mat<double, numNodes, numStates> localNodeStates;
    A2D::Mat<double, numNodes, numDim> localNodeCoords;
    A2D::Mat<double, numNodes, numStates> localRes;
    gatherElementData<numNodes, numStates, numDim>(connPtr,
                                                   conn,
                                                   elementInd,
                                                   states,
                                                   nodeCoords,
                                                   localNodeStates,
                                                   localNodeCoords);

    // Now the main quadrature loop
    for (int quadPtXInd = 0; quadPtXInd < (order + 1); quadPtXInd++) {
      const double quadPtXWeight = getGaussQuadWeight<order>(quadPtXInd);
      const double quadPtXCoord = getGaussQuadCoord<order>(quadPtXInd);
      for (int quadPtYInd = 0; quadPtYInd < (order + 1); quadPtYInd++) {
        const double quadPtYWeight = getGaussQuadWeight<order>(quadPtYInd);
        const double quadPtYCoord = getGaussQuadCoord<order>(quadPtYInd);
        const double quadPtWeight = quadPtXWeight * quadPtYWeight;
        const double quadPtXi[2] = {quadPtXCoord, quadPtYCoord};

        // Compute Jacobian, J = dx/dxi
        A2D::Mat<double, numDim, numDim> J, JInv;
        interpParamGradient<order, numDim, numDim>(quadPtXi, localNodeCoords, J);

        // --- Compute J^-1 and detJ ---
        A2D::MatInv(J, JInv);
        double detJ;
        A2D::MatDet(J, detJ);

        // --- Compute state gradient in physical space ---
        A2D::Mat<double, numStates, numDim> dudx;
        interpRealGradient(quadPtXi, localNodeStates, JInv, dudx);

        // Compute weak residual integrand (derivative of energy w.r.t state gradient scaled by quadrature weight and
        // detJ)
        A2D::Mat<double, numStates, numDim> weakRes;
        planeStressWeakRes<strainType>(dudx, E, nu, t, quadPtWeight * detJ, weakRes);

        // Add to residual (transform sensitivity to be w.r.t nodal states and scale by quadrature weight and detJ)
        addTransformStateGradSens<order, numStates, numDim>(quadPtXi, weakRes, JInv, localRes);
      }
    }
    // --- Scatter local residual back to global array---
    scatterResidual(connPtr, conn, elementInd, localRes, residual);
  }
}

template <int order, int numStates, int numDim, A2D::GreenStrainType strainType = A2D::GreenStrainType::LINEAR>
void assemblePlaneStressJacobian(const int *const connPtr,
                                 const int *const conn,
                                 const int numElements,
                                 const double *const states,
                                 const double *const nodeCoords,
                                 const double E,
                                 const double nu,
                                 const double t,
                                 int *bcsrMap,
                                 double *const residual,
                                 double *const matEntries) {
  constexpr int numNodes = (order + 1) * (order + 1);
#pragma omp parallel for
  for (int elementInd = 0; elementInd < numElements; elementInd++) {
    // Get the element nodal states and coordinates
    A2D::Mat<double, numNodes, numStates> localNodeStates;
    A2D::Mat<double, numNodes, numDim> localNodeCoords;
    gatherElementData<numNodes, numStates, numDim>(connPtr,
                                                   conn,
                                                   elementInd,
                                                   states,
                                                   nodeCoords,
                                                   localNodeStates,
                                                   localNodeCoords);

    A2D::Mat<double, numNodes, numStates> localRes;
    const int numDOF = numNodes * numStates;
    A2D::Mat<double, numDOF, numDOF> localMat;

    // Now the main quadrature loop
    for (int quadPtXInd = 0; quadPtXInd < (order + 1); quadPtXInd++) {
      const double quadPtXWeight = getGaussQuadWeight<order>(quadPtXInd);
      const double quadPtXCoord = getGaussQuadCoord<order>(quadPtXInd);
      for (int quadPtYInd = 0; quadPtYInd < (order + 1); quadPtYInd++) {
        const double quadPtYWeight = getGaussQuadWeight<order>(quadPtYInd);
        const double quadPtYCoord = getGaussQuadCoord<order>(quadPtYInd);
        const double quadPtWeight = quadPtXWeight * quadPtYWeight;
        const double quadPtXi[2] = {quadPtXCoord, quadPtYCoord};

        // Compute Jacobian, J = dx/dxi
        A2D::Mat<double, numDim, numDim> J, JInv;
        interpParamGradient<order, numDim, numDim>(quadPtXi, localNodeCoords, J);

        // --- Compute J^-1 and detJ ---
        A2D::MatInv(J, JInv);
        double detJ;
        A2D::MatDet(J, detJ);

        // --- Compute state gradient in physical space ---
        A2D::Mat<double, numStates, numDim> dudx;
        interpRealGradient(quadPtXi, localNodeStates, JInv, dudx);

        // Compute weak residual integrand (derivative of energy w.r.t state gradient scaled by quadrature weight and
        // detJ)
        A2D::Mat<double, numStates, numDim> weakRes;
        planeStressWeakRes<strainType>(dudx, E, nu, t, quadPtWeight * detJ, weakRes);

        // Add to residual (transform sensitivity to be w.r.t nodal states and scale by quadrature weight and detJ)
        addTransformStateGradSens<order, numStates, numDim>(quadPtXi, weakRes, JInv, localRes);

        // We will compute the element Jac one column at a time, essentially doing a forward AD pass through the
        // residual calculation

        // Create a matrix of AD scalars that will store dudx and it's forward seed
        A2D::Mat<A2D::ADScalar<double, 1>, numStates, numDim> dudxFwd;
        for (int ii = 0; ii < numStates; ii++) {
          for (int jj = 0; jj < numDim; jj++) {
            dudxFwd(ii, jj).value = dudx(ii, jj);
          }
        }
        for (int nodeYInd = 0; nodeYInd < (order + 1); nodeYInd++) {
          for (int nodeXInd = 0; nodeXInd < (order + 1); nodeXInd++) {
            const int nodeInd = nodeYInd * (order + 1) + nodeXInd;
            double N;
            // Technically we should set a forward seed of 1 in q(nodeInd, stateInd) and 0 in all other entries, then
            // propogate that seed through the state gradient interpolation, but because we are only seeding a single
            // nodal state each round, we only need to use the basis function gradient for that node to propogate
            // through the state gradient calculation

            // Forward seed of dudxi is just dNdxi for this node
            A2D::Mat<double, 1, numDim> dudxiDot, dudxDot;
            lagrangePoly2dDeriv<double, order>(quadPtXi, nodeXInd, nodeYInd, N, dudxiDot);
            // Now propogate through dudx = dudxi * J^-1
            A2D::MatMatMult(dudxiDot, JInv, dudxDot);
            for (int stateInd = 0; stateInd < numStates; stateInd++) {
              // Now we will do a forward AD pass through the weak residual calculation by setting dudxDot as the seed
              // in the state gradient
              dudxFwd.zero();
              for (int jj = 0; jj < numDim; jj++) {
                dudxFwd(stateInd, jj).deriv[0] = dudxDot(stateInd, jj);
              }
              A2D::Mat<A2D::ADScalar<double, 1>, numStates, numDim> weakResFwd;
              planeStressWeakRes<strainType>(dudxFwd, E, nu, t, quadPtWeight * detJ, weakResFwd);

              // Put the forward seed of the weak residual into it's own matrix
              A2D::Mat<double, numStates, numDim> weakResDot;
              for (int ii = 0; ii < numStates; ii++) {
                for (int jj = 0; jj < numDim; jj++) {
                  weakResDot(ii, jj) = weakResFwd(ii, jj).deriv[0];
                }
              }

              // Now propogate through the transformation of the state gradient sensitivity, this gives us this quad
              // point's contribution to this column of the element jacobian
              A2D::Mat<double, numNodes, numStates> matColContribution;
              addTransformStateGradSens<order, numStates, numDim>(quadPtXi, weakResDot, JInv, matColContribution);

              // Add this column contribution to the element jacobian
              const int colInd = nodeInd * numStates + stateInd;
              for (int ii = 0; ii < numNodes; ii++) {
                for (int jj = 0; jj < numStates; jj++) {
                  const int rowInd = ii * numStates + jj;
                  localMat(colInd, rowInd) += matColContribution(ii, jj);
                }
              }
            }
          }
        }
      }
    }
    // --- Scatter the local element Jacobian into the global matrix ---
    scatterMat<numNodes, numStates>(elementInd, &bcsrMap[elementInd * numNodes * numNodes], localMat, matEntries);

    // --- Scatter local residual back to global array---
    if (residual != nullptr) {
      scatterResidual(connPtr, conn, elementInd, localRes, residual);
    }
  }
}

// #ifdef __CUDACC__
template <int numElements, int numNodes, int valsPerNode>
__device__ void
gatherElementNodalValues(const int *const connPtr,
                         const int *const conn,
                         const double *const globalData,
                         const int threadID,
                         const int threadBlockSize,
                         const int firstElemGlobalInd,
                         const int elementsToLoad, // Need this in case out block goes past the last element
                         double elemData[numElements][numNodes * valsPerNode]) {
  const int valsPerElem = numNodes * valsPerNode;

  for (int ii = threadID; ii < elementsToLoad * valsPerElem; ii += threadBlockSize) {
    const int elemLoadInd = ii / valsPerElem; // Which element am I loading data for
    const int globalElemLoadInd =
        firstElemGlobalInd + elemLoadInd;                // What's the global index of the element I'm loading data for
    const int arrayLoadInd = ii % valsPerElem;           // Which entry of the array for this element am I loading
    const int nodeLoadInd = arrayLoadInd / valsPerNode;  // Which node within the element am I loading data for
    const int stateLoadInd = arrayLoadInd % valsPerNode; // Which state am I loading for this node?
    const int globalNodeLoadInd =
        conn[connPtr[globalElemLoadInd] + nodeLoadInd]; // What's the global ID of the node I'm loading data for

    const int globalDataInd = globalNodeLoadInd * valsPerNode + stateLoadInd;

#ifndef NDEBUG
    printf("thread %d loading value %d for node %d in element %d, from index %d in global vector\n",
           threadID,
           stateLoadInd,
           nodeLoadInd,
           globalElemLoadInd,
           globalDataInd);
#endif

    elemData[elemLoadInd][nodeLoadInd * valsPerNode + stateLoadInd] = globalData[globalDataInd];
  }
}

template <int numElements, int numNodes, int valsPerNode>
__device__ void
scatterElementNodalValues(const int *const connPtr,
                          const int *const conn,
                          const int threadID,
                          const int threadBlockSize,
                          const int firstElemGlobalInd,
                          const int elementsToWrite, // Need this in case out block goes past the last element
                          const double elemData[numElements][numNodes * valsPerNode],
                          double *globalData) {
  const int valsPerElem = numNodes * valsPerNode;

  for (int ii = threadID; ii < elementsToWrite * valsPerElem; ii += threadBlockSize) {
    const int elemWriteInd = ii / valsPerElem; // Which element am I writing data for
    const int globalElemWriteInd =
        firstElemGlobalInd + elemWriteInd;                // What's the global index of the element I'm writing data for
    const int arrayWriteInd = ii % valsPerElem;           // Which entry of the array for this element am I writing
    const int nodeWriteInd = arrayWriteInd / valsPerNode; // Which node within the element am I writing data for
    const int stateWriteInd = arrayWriteInd % valsPerNode; // Which state am I writing for this node?
    const int globalNodeWriteInd =
        conn[connPtr[globalElemWriteInd] + nodeWriteInd]; // What's the global ID of the node I'm writing data for

    const int globalDataInd = globalNodeWriteInd * valsPerNode + stateWriteInd;

#ifndef NDEBUG
    printf("thread %d writing value %d from node %d in element %d, to index %d in global vector\n",
           threadID,
           stateWriteInd,
           nodeWriteInd,
           globalElemWriteInd,
           globalDataInd);
#endif

    atomicAdd(&globalData[globalDataInd], elemData[elemWriteInd][nodeWriteInd * valsPerNode + stateWriteInd]);
  }
}

template <int order,
          int numStates,
          int numDim,
          int elemPerBlock = 1,
          A2D::GreenStrainType strainType = A2D::GreenStrainType::LINEAR>
__global__ void assemblePlaneStressResidualKernel(const int *const connPtr,
                                                  const int *const conn,
                                                  const int numElements,
                                                  const double *const states,
                                                  const double *const nodeCoords,
                                                  const double E,
                                                  const double nu,
                                                  const double t,
                                                  double *const residual) {
  constexpr int numNodes = (order + 1) * (order + 1);
  constexpr int numQuadPts = numNodes;

  // Figure out various thread indices, one thread per quad point
  const int blockSize = blockDim.x;
  const int localThreadInd = threadIdx.x;
  const int localElementInd = localThreadInd / numQuadPts;
  const int globalElementInd = elemPerBlock * blockIdx.x + localElementInd;
  const int quadPtInd = localThreadInd % numQuadPts;

  // ==============================================================================
  // Load element data
  // ==============================================================================
  // Local element data will live in shared memory as arrays of A2Dmats
  __shared__ double localNodeStates[elemPerBlock][numNodes * numStates];
  __shared__ double localNodeCoords[elemPerBlock][numNodes * numDim];
  __shared__ double localRes[elemPerBlock][numNodes * numStates];

#ifndef NDEBUG
  printf("assemblePlaneStressResidualKernel: Created shared memory\n");
#endif

  // zero the local residual
  for (int ii = threadIdx.x; ii < (elemPerBlock * numNodes * numStates); ii += blockDim.x) {
    const int DOFPerElem = numNodes * numStates;
    const int eInd = ii / DOFPerElem;
    const int arrInd = ii % DOFPerElem;
    localRes[eInd][arrInd] = 0.0;
  }

  // Before loading data we need to whether this block will go past the last element and reign it in if it will
  const int blockStartElement = elemPerBlock * blockIdx.x;
  const int numElementsToCompute =
      blockStartElement + elemPerBlock >= numElements ? numElements - blockStartElement : elemPerBlock;

#ifndef NDEBUG
  if (numElementsToCompute < elemPerBlock) {
    printf("Thread block %d: Supposed to process %d elements, but limited to %d as reached last element\n",
           blockIdx.x,
           elemPerBlock,
           numElementsToCompute);
  }
#endif

  gatherElementNodalValues<elemPerBlock, numNodes, numStates>(connPtr,
                                                              conn,
                                                              states,
                                                              localThreadInd,
                                                              blockSize,
                                                              blockStartElement,
                                                              numElementsToCompute,
                                                              localNodeStates);

  gatherElementNodalValues<elemPerBlock, numNodes, numDim>(connPtr,
                                                           conn,
                                                           nodeCoords,
                                                           localThreadInd,
                                                           blockSize,
                                                           blockStartElement,
                                                           numElementsToCompute,
                                                           localNodeCoords);

#ifndef NDEBUG
  printf("assemblePlaneStressResidualKernel: Finished loading element data\n");
#endif

  __syncthreads();

  const bool isActiveThread = globalElementInd < numElements && localElementInd < elemPerBlock;

  if (isActiveThread) {

    // Now the main quadrature loop
    const int quadPtXInd = quadPtInd % (order + 1);
    const int quadPtYInd = quadPtInd / (order + 1);
    const double quadPtWeight = getGaussQuadWeight<order>(quadPtXInd) * getGaussQuadWeight<order>(quadPtYInd);
    const double quadPtXi[2] = {getGaussQuadCoord<order>(quadPtXInd), getGaussQuadCoord<order>(quadPtYInd)};

    // Compute Jacobian, J = dx/dxi
    A2D::Mat<double, numDim, numDim> J, JInv;
    interpParamGradient<order, numDim, numDim>(quadPtXi, localNodeCoords[localElementInd], J);

#ifndef NDEBUG
    printf("assemblePlaneStressResidualKernel: Finished interpParamGradient\n");
#endif

    // --- Compute J^-1 and detJ ---
    A2D::MatInv(J, JInv);
    double detJ;
    A2D::MatDet(J, detJ);

    // --- Compute state gradient in physical space ---
    A2D::Mat<double, numStates, numDim> dudx;
    interpRealGradient<order, numDim, numDim>(quadPtXi, localNodeStates[localElementInd], JInv, dudx);

#ifndef NDEBUG
    printf("assemblePlaneStressResidualKernel: Finished interpRealGradient\n");
#endif

    // Compute weak residual integrand (derivative of energy w.r.t state gradient scaled by quadrature weight and
    // detJ)
    A2D::Mat<double, numStates, numDim> weakRes;
    planeStressWeakRes<strainType>(dudx, E, nu, t, quadPtWeight * detJ, weakRes);

#ifndef NDEBUG
    printf("assemblePlaneStressResidualKernel: Finished planeStressWeakRes\n");
#endif

    // Add to residual (transform sensitivity to be w.r.t nodal states and scale by quadrature weight and detJ)
    addTransformStateGradSens<order, numStates, numDim, true>(quadPtXi, weakRes, JInv, localRes[localElementInd]);

#ifndef NDEBUG
    printf("assemblePlaneStressResidualKernel: Finished addTransformStateGradSens\n");
#endif
  }
  __syncthreads();
  // --- Scatter local residual back to global array---
  scatterElementNodalValues<elemPerBlock, numNodes, numStates>(connPtr,
                                                               conn,
                                                               localThreadInd,
                                                               blockSize,
                                                               blockStartElement,
                                                               numElementsToCompute,
                                                               localRes,
                                                               residual);

#ifndef NDEBUG
  printf("assemblePlaneStressResidualKernel: Finished scatterElementNodalValues\n");
#endif
}

template <int numElements, int numNodes, int valsPerNode>
__device__ void
scatterElementMat(const int *const map,
                  const int threadID,
                  const int threadBlockSize,
                  const int firstElemGlobalInd,
                  const int elementsToWrite, // Need this in case out block goes past the last element
                  const double elemMat[numElements][(numNodes * valsPerNode) * (numNodes * valsPerNode)],
                  double *const globalMatData) {
  constexpr int nodesPerElem = numNodes * numNodes;
  constexpr int numDOF = numNodes * valsPerNode;
  constexpr int valsPerElem = numDOF * numDOF;
  constexpr int matBlockSize = valsPerNode * valsPerNode;

  for (int ii = threadID; ii < elementsToWrite * valsPerElem; ii += threadBlockSize) {
    const int elemWriteInd = ii / valsPerElem; // Which element am I writing data for
    const int globalElemWriteInd =
        firstElemGlobalInd + elemWriteInd; // What's the global index of the element I'm writing data for

    // Now we have a decision to make, do we assign consecutive threads to consecutive entries in the element matrix?
    // Or to consecutive entries in each global matrix block. The first approach will lead to more coalesced reads,
    // the second should lead to more coalesced writes. For now I'll assign consecutive threads to consecutive entries
    // in each block

    const int elemInternalInd = ii % valsPerElem;                // What's my index within this element?
    const int blockInd = elemInternalInd / matBlockSize;         // Which block of the element matrix am I handling?
    const int blockInternalInd = elemInternalInd % matBlockSize; // Which entry within the block am I handling
    // What are the node indices for the block I'm handling
    const int blockColInd = blockInd % numNodes;
    const int blockRowInd = blockInd / numNodes;
    const int elemMatColInd = blockColInd * valsPerNode + blockInternalInd % valsPerNode;
    const int elemMatRowInd = blockRowInd * valsPerNode + blockInternalInd / valsPerNode;
    const int elemMatFlatInd = elemMatRowInd * numDOF + elemMatColInd;

    const int mapInd = globalElemWriteInd * nodesPerElem + blockRowInd * numNodes + blockColInd;
    const int blockStartInd = map[mapInd];

#ifndef NDEBUG
    printf("Thread %d: scattering element %d mat[ %d, %d] to entry %d in block starting at entry %d in global mat "
           "data\n",
           threadID,
           globalElemWriteInd,
           elemMatRowInd,
           elemMatColInd,
           blockInternalInd,
           blockStartInd);
#endif

    atomicAdd(&globalMatData[blockStartInd + blockInternalInd], elemMat[elemWriteInd][elemMatFlatInd]);
  }
}

template <int order,
          int numStates,
          int numDim,
          int elemPerBlock = 1,
          A2D::GreenStrainType strainType = A2D::GreenStrainType::LINEAR>
__global__ void assemblePlaneStressJacobianKernel(const int *const connPtr,
                                                  const int *const conn,
                                                  const int numElements,
                                                  const double *const states,
                                                  const double *const nodeCoords,
                                                  const double E,
                                                  const double nu,
                                                  const double t,
                                                  int *bcsrMap,
                                                  double *const residual,
                                                  double *const matEntries) {
  constexpr int numNodes = (order + 1) * (order + 1);
  constexpr int numDOF = numNodes * numStates;
  constexpr int elementMatSize = numDOF * numDOF;

  // Figure out various thread indices, one thread per mat column
  const int blockSize = blockDim.x;
  const int localThreadInd = threadIdx.x;
  const int localElementInd = localThreadInd / numDOF;
  // const int globalElementInd = elemPerBlock * blockIdx.x + localElementInd;

  const int columnInd = localThreadInd % numDOF;

  // ==============================================================================
  // Load element data
  // ==============================================================================
  // Local element data will live in shared memory as arrays of A2Dmats
  __shared__ double localNodeStates[elemPerBlock][numDOF];
  __shared__ double localNodeCoords[elemPerBlock][numNodes * numDim];
  __shared__ double localRes[elemPerBlock][numDOF];
  __shared__ double localMat[elemPerBlock][elementMatSize];

#ifndef NDEBUG
  printf("assemblePlaneStressJacobianKernel: Created shared memory\n");
#endif

  // zero the local residual
  for (int ii = localThreadInd; ii < (elemPerBlock * numDOF); ii += blockDim.x) {
    const int eInd = ii / numDOF;
    const int arrInd = ii % numDOF;
    localRes[eInd][arrInd] = 0.0;
  }

  // Zero the local matrix
  for (int ii = localThreadInd; ii < (elemPerBlock * elementMatSize); ii += blockDim.x) {
    const int eInd = ii / elementMatSize;
    const int arrInd = ii % elementMatSize;
    localMat[eInd][arrInd] = 0.0;
  }

#ifndef NDEBUG
  printf("assemblePlaneStressJacobianKernel: Zeroed element matrices\n");
#endif

  // Before loading data we need to whether this block will go past the last element and reign it in if it will
  const int blockStartElement = elemPerBlock * blockIdx.x;
  const int numElementsToCompute =
      blockStartElement + elemPerBlock >= numElements ? numElements - blockStartElement : elemPerBlock;

#ifndef NDEBUG
  if (numElementsToCompute < elemPerBlock) {
    printf("Thread block %d: Supposed to process %d elements, but limited to %d as reached last element\n",
           blockIdx.x,
           elemPerBlock,
           numElementsToCompute);
  }
#endif

  // Get the element nodal states and coordinates using coalesced loads
  gatherElementNodalValues<elemPerBlock, numNodes, numStates>(connPtr,
                                                              conn,
                                                              states,
                                                              localThreadInd,
                                                              blockSize,
                                                              blockStartElement,
                                                              numElementsToCompute,
                                                              localNodeStates);

  gatherElementNodalValues<elemPerBlock, numNodes, numDim>(connPtr,
                                                           conn,
                                                           nodeCoords,
                                                           localThreadInd,
                                                           blockSize,
                                                           blockStartElement,
                                                           numElementsToCompute,
                                                           localNodeCoords);

  __syncthreads();

#ifndef NDEBUG
  printf("assemblePlaneStressJacobianKernel: Finished loading element data\n");
#endif

  const bool isActiveThread = localElementInd < numElementsToCompute;

  // Now the main quadrature loop
  if (isActiveThread) {
    for (int quadPtXInd = 0; quadPtXInd < (order + 1); quadPtXInd++) {
      const double quadPtXWeight = getGaussQuadWeight<order>(quadPtXInd);
      const double quadPtXCoord = getGaussQuadCoord<order>(quadPtXInd);
      for (int quadPtYInd = 0; quadPtYInd < (order + 1); quadPtYInd++) {
        const double quadPtYWeight = getGaussQuadWeight<order>(quadPtYInd);
        const double quadPtYCoord = getGaussQuadCoord<order>(quadPtYInd);
        const double quadPtWeight = quadPtXWeight * quadPtYWeight;
        const double quadPtXi[2] = {quadPtXCoord, quadPtYCoord};

        // We will compute the element Jac one column at a time, essentially doing a forward AD pass through the
        // residual calculation:
        // Technically we should set a forward seed of 1 in q(nodeInd, stateInd) and 0 in all other entries, then
        // propogate that seed through the state gradient interpolation, but because we are only seeding a single
        // nodal state, we only need to use the basis function gradient for that node to propogate through the state
        // gradient calculation
        const int nodeInd = columnInd / numStates;
        const int nodeXInd = nodeInd % (order + 1);
        const int nodeYInd = nodeInd / (order + 1);
        const int stateInd = columnInd % numStates;
        double N;

        // Compute Jacobian, J = dx/dxi
        A2D::Mat<double, numDim, numDim> J, JInv;
        interpParamGradient<order, numDim, numDim>(quadPtXi, localNodeCoords[localElementInd], J);

        // --- Compute J^-1 and detJ ---
        A2D::MatInv(J, JInv);
        double detJ;
        A2D::MatDet(J, detJ);

        // Forward seed of dudxi is just dNdxi for this node
        A2D::Mat<double, 1, numDim> dudxiDot, dudxDot;
        lagrangePoly2dDeriv<double, order>(quadPtXi, nodeXInd, nodeYInd, N, dudxiDot.get_data());

        // Compute Jacobian, J = dx/dxi
        A2D::Mat<double, numDim, numDim> J, JInv;
        interpParamGradient<order, numDim, numDim>(quadPtXi, localNodeCoords[localElementInd], J);

        // --- Compute state gradient in physical space ---
        A2D::Mat<double, numStates, numDim> dudx;
        interpRealGradient<order, numDim, numDim>(quadPtXi, localNodeStates[localElementInd], JInv, dudx);

        // Now propogate through dudx = dudxi * J^-1
        A2D::MatMatMult(dudxiDot, JInv, dudxDot);

        // Create a matrix of AD scalars that will store dudx and it's forward seed
        A2D::Mat<A2D::ADScalar<double, 1>, numStates, numDim> dudxFwd;
        for (int ii = 0; ii < numStates; ii++) {
          for (int jj = 0; jj < numDim; jj++) {
            dudxFwd(ii, jj).value = dudx(ii, jj);
          }
        }

        // Now we will do a forward AD pass through the weak residual calculation by
        // setting dudxDot as the seed in the state gradient
        for (int jj = 0; jj < numDim; jj++) {
          dudxFwd(stateInd, jj).deriv[0] = dudxDot[jj];
        }

        A2D::Mat<A2D::ADScalar<double, 1>, numStates, numDim> weakResFwd;
        planeStressWeakRes<strainType>(dudxFwd, E, nu, t, quadPtWeight * detJ, weakResFwd);

        // We now have the weak residual value, and it's forward seed in weakResFwd, we need to extract the value and
        // the forward seed into separate arrays into to compute the residual and jacobian entries
        A2D::Mat<double, numStates, numDim> weakRes, weakResDot;
        for (int ii = 0; ii < numStates; ii++) {
          for (int jj = 0; jj < numDim; jj++) {
            weakRes(ii, jj) = weakResFwd(ii, jj).value;
            weakResDot(ii, jj) = weakResFwd(ii, jj).deriv[0];
          }
        }

        // On the thread computing the ith column of the jacobian, we will contribute only to the ith entry of the
        // residual by mapping the weak residual (the sensitivity of the strain energy w.r.t the state gradient) to the
        // sensitivity of the strain energy w.r.t that single DOF
        addTransformStateGradSens<order, numStates, numDim>(quadPtXi,
                                                            weakRes,
                                                            JInv,
                                                            nodeXInd,
                                                            nodeYInd,
                                                            stateInd,
                                                            localRes[localElementInd][columnInd]);

        // Now propogate through the transformation of the state gradient sensitivity, this gives us this quad point's
        // contribution to this column of the element jacobian
        double matColContribution[numDOF];
        memset(matColContribution, 0, numDOF * sizeof(double));
        addTransformStateGradSens<order, numStates, numDim>(quadPtXi, weakResDot, JInv, matColContribution);

        // #ifndef NDEBUG
        //         printf("assemblePlaneStressJacobianKernel: Finished addTransformStateGradSens\n");
        // #endif

        // Add this column contribution to the element jacobian, consecutive threads should be adding to consecutive
        // entries in the row of the local matrix, which should give good performance?
        for (int ii = 0; ii < numDOF; ii++) {
          localMat[localElementInd][ii * numDOF + columnInd] += matColContribution[ii];
        }
        // #ifndef NDEBUG
        // printf("assemblePlaneStressJacobianKernel: Finished adding matColContribution to localMat\n");
        // printf("Thread %d: Quad pt [%d, %d] added %f to row 0, column %d, value is now %f\n",
        //        localThreadInd,
        //        quadPtXInd,
        //        quadPtYInd,
        //        matColContribution[0],
        //        columnInd,
        //        localMat[localElementInd][columnInd]);
        // #endif
      }
    }

#ifndef NDEBUG
    if (columnInd == 0) {
      printf("Element %d Jacobian = \n", localElementInd);
      printMat(localMat[localElementInd], numDOF, numDOF);
    }
#endif
  }
  __syncthreads();
  // --- Scatter the local element Jacobian into the global matrix ---
  scatterElementMat<elemPerBlock, numNodes, numStates>(bcsrMap,
                                                       localThreadInd,
                                                       blockSize,
                                                       blockStartElement,
                                                       numElementsToCompute,
                                                       localMat,
                                                       matEntries);

  // --- Scatter local residual back to global array---
  if (residual != nullptr) {
    scatterElementNodalValues<elemPerBlock, numNodes, numStates>(connPtr,
                                                                 conn,
                                                                 localThreadInd,
                                                                 blockSize,
                                                                 blockStartElement,
                                                                 numElementsToCompute,
                                                                 localRes,
                                                                 residual);
  }
}
// #endif

constexpr int getResidualElemPerBlock(const int elementOrder) {
  switch (elementOrder) {
    case 1:
      return 8;
      break;
    case 2:
      return 7; // 7 best so far 5.17809000000e-04
      break;
    case 3:
      return 2;
      break;
    case 4:
      return 5; // 5 best so far
      break;

    default:
      return 8;
      break;
  }
}

double runResidualKernel(const int elementOrder,
                         const int *const connPtr,
                         const int *const conn,
                         const int numElements,
                         const double *const states,
                         const double *const nodeCoords,
                         const double E,
                         const double nu,
                         const double t,
                         double *const residual) {
#ifdef __CUDACC__
  // Figure out how many blocks and threads to use
  // const int elemPerBlock = 8;
  // const int numQuadPoints = (elementOrder + 1) * (elementOrder + 1);
  // const int warpsPerBlock = (elemPerBlock * numQuadPoints + 32 - 1) / 32;
  // const int threadsPerBlock = 32 * warpsPerBlock;
  // const int numBlocks = (numElements + elemPerBlock - 1) / elemPerBlock;
// #ifndef NDEBUG
//   printf("For %d elements, launching %d blocks with %d threads each, %d elements per block\n",
//          numElements,
//          numBlocks,
//          threadsPerBlock,
//          elemPerBlock);
// #endif
#endif
  auto t1 = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
#define ASSEMBLE_PLANE_STRESS_RESIDUAL(elementOrder)                                                                   \
  const int elemPerBlock = getResidualElemPerBlock(elementOrder);                                                      \
  const int numQuadPoints = (elementOrder + 1) * (elementOrder + 1);                                                   \
  const int warpsPerBlock = (elemPerBlock * numQuadPoints + 32 - 1) / 32;                                              \
  const int threadsPerBlock = 32 * warpsPerBlock;                                                                      \
  const int numBlocks = (numElements + elemPerBlock - 1) / elemPerBlock;                                               \
  assemblePlaneStressResidualKernel<elementOrder, 2, 2, elemPerBlock>                                                  \
      <<<numBlocks, threadsPerBlock>>>(connPtr, conn, numElements, states, nodeCoords, E, nu, t, residual);
#else
#define ASSEMBLE_PLANE_STRESS_RESIDUAL(elementOrder)                                                                   \
  assemblePlaneStressResidual<elementOrder, 2, 2, elemPerBlock>(connPtr,                                               \
                                                                conn,                                                  \
                                                                numElements,                                           \
                                                                states,                                                \
                                                                nodeCoords,                                            \
                                                                E,                                                     \
                                                                nu,                                                    \
                                                                t,                                                     \
                                                                residual);
#endif

  switch (elementOrder) {
    case 1: {
      ASSEMBLE_PLANE_STRESS_RESIDUAL(1);
    } break;
    case 2: {
      ASSEMBLE_PLANE_STRESS_RESIDUAL(2);
    } break;
    case 3: {
      ASSEMBLE_PLANE_STRESS_RESIDUAL(3);
    } break;
    case 4: {
      ASSEMBLE_PLANE_STRESS_RESIDUAL(4);
    } break;
    default:
      break;
  }
#ifdef __CUDACC__
  gpuErrchk(cudaDeviceSynchronize());
#endif
  auto t2 = std::chrono::high_resolution_clock::now();
  /* Getting number of seconds as a double. */
  std::chrono::duration<double> tmp = t2 - t1;
  return tmp.count();
}

constexpr int getJacobianElemPerBlock(const int elementOrder) {
  switch (elementOrder) {
    case 1:
      return 16; // 16 best so far: 1.98348797858e-03
      break;
    case 2:
      return 7; // 7 best so far: 3.42732807621e-03, 15 returns all zeroes for some reason
      break;
    case 3:
      return 1; // 1 best so far: 6.99187163264e-03, 6 and above uses too much shared mem
      break;
    case 4:
      return 2; // 2 best so far: 1.47015675902e-02, 3 and above uses too much shared memory
      break;

    default:
      return 2;
      break;
  }
}

double runJacobianKernel(const int elementOrder,
                         const int *const connPtr,
                         const int *const conn,
                         const int numElements,
                         const double *const states,
                         const double *const nodeCoords,
                         const double E,
                         const double nu,
                         const double t,
                         int *elementBCSRMap,
                         double *const residual,
                         double *const matEntries) {
#ifdef __CUDACC__
  // Figure out how many blocks and threads to use
  //   const int elemPerBlock = 2;
  //   const int numDOF = 2 * (elementOrder + 1) * (elementOrder + 1);
  //   const int warpsPerBlock = (elemPerBlock * numDOF + 32 - 1) / 32;
  //   const int threadsPerBlock = 32 * warpsPerBlock;
  //   const int numBlocks = (numElements + elemPerBlock - 1) / elemPerBlock;
  // #ifndef NDEBUG
  //   printf("For %d elements, launching %d blocks with %d threads each, %d elements per block\n",
  //          numElements,
  //          numBlocks,
  //          threadsPerBlock,
  //          elemPerBlock);
  // #endif
  // --- Create timing events ---
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
#else
  auto t1 = std::chrono::high_resolution_clock::now();
#endif

// Helper macro so I don't have to write out all these inputs every time
#ifdef __CUDACC__
#define ASSEMBLE_PLANE_STRESS_JACOBIAN(elementOrder)                                                                   \
  const int elemPerBlock = getJacobianElemPerBlock(elementOrder);                                                      \
  const int numDOF = 2 * (elementOrder + 1) * (elementOrder + 1);                                                      \
  const int warpsPerBlock = (elemPerBlock * numDOF + 32 - 1) / 32;                                                     \
  const int threadsPerBlock = 32 * warpsPerBlock;                                                                      \
  const int numBlocks = (numElements + elemPerBlock - 1) / elemPerBlock;                                               \
  assemblePlaneStressJacobianKernel<elementOrder, 2, 2, elemPerBlock><<<numBlocks, threadsPerBlock>>>(connPtr,         \
                                                                                                      conn,            \
                                                                                                      numElements,     \
                                                                                                      states,          \
                                                                                                      nodeCoords,      \
                                                                                                      E,               \
                                                                                                      nu,              \
                                                                                                      t,               \
                                                                                                      elementBCSRMap,  \
                                                                                                      residual,        \
                                                                                                      matEntries);
#else
#define ASSEMBLE_PLANE_STRESS_JACOBIAN(elementOrder)                                                                   \
  assemblePlaneStressJacobian<elementOrder, 2, 2>(connPtr,                                                             \
                                                  conn,                                                                \
                                                  numElements,                                                         \
                                                  states,                                                              \
                                                  nodeCoords,                                                          \
                                                  E,                                                                   \
                                                  nu,                                                                  \
                                                  t,                                                                   \
                                                  elementBCSRMap,                                                      \
                                                  residual,                                                            \
                                                  matEntries);
#endif
  // We need a switch statement here because the kernel is templated on the number of nodes, which we only
  // know at runtime
  switch (elementOrder) {
    case 1: {
      ASSEMBLE_PLANE_STRESS_JACOBIAN(1);
    } break;
    case 2: {
      ASSEMBLE_PLANE_STRESS_JACOBIAN(2);
    } break;
    case 3: {
      ASSEMBLE_PLANE_STRESS_JACOBIAN(3);
    } break;
    case 4: {
      ASSEMBLE_PLANE_STRESS_JACOBIAN(4);
    } break;
    default:
      break;
  }
#ifdef __CUDACC__
  gpuErrchk(cudaDeviceSynchronize());
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float runTime;
  cudaEventElapsedTime(&runTime, start, stop);
  runTime /= 1000; // Convert to seconds
  return double(runTime);
#else
  auto t2 = std::chrono::high_resolution_clock::now();
  /* Getting number of seconds as a double. */
  std::chrono::duration<double> tmp = t2 - t1;
  return tmp.count();
#endif
}
