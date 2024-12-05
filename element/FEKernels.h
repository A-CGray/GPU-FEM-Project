#pragma once

#include "GPUMacros.h"
#include "GaussQuadrature.h"
#include "LagrangeShapeFuncs.h"
#include "a2dcore.h"
#include "adscalar.h"

template <typename T>
__HOST_AND_DEVICE__ void printMat(const T mat) {
  for (int ii = 0; ii < mat.nrows; ii++) {
    printf("[");
    for (int jj = 0; jj < mat.ncols; jj++) {
      printf(" % f, ", mat(ii, jj));
    }
    printf("]\n");
  }
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
__DEVICE__ void scatterMat(const int *const connPtr,
                           const int *const conn,
                           const int elementInd,
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

template <int order, int numVals, int numDim>
__DEVICE__ void addTransformStateGradSens(const double xi[numDim],
                                          const A2D::Mat<double, numVals, numDim> stateGradSens,
                                          const A2D::Mat<double, numDim, numDim> JInv,
                                          A2D::Mat<double, (order + 1) * (order + 1), numVals> &nodalValSens) {
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

        for (int kk = 0; kk < numDim; kk++) {
          for (int jj = 0; jj < numVals; jj++) {
            nodalValSens(nodeInd, jj) += dNidxi[kk] * temp(kk, jj);
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
        for (int kk = 0; kk < numDim; kk++) {
          nodalValSens(ii, jj) += temp(ii, kk) * stateGradSens(jj, kk);
        }
      }
    }
  }
}

// dudxi = localNodeValues^T * dNdxi
template <int order, int numVals, int numDim>
__DEVICE__ void interpParamGradient(const double xi[numDim],
                                    const A2D::Mat<double, (order + 1) * (order + 1), numVals> nodalValues,
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
                                   const A2D::Mat<double, (order + 1) * (order + 1), numVals> nodalValues,
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
    scatterMat<numNodes, numStates>(connPtr,
                                    conn,
                                    elementInd,
                                    &bcsrMap[elementInd * numNodes * numNodes],
                                    localMat,
                                    matEntries);

    // --- Scatter local residual back to global array---
    if (residual != nullptr) {
      scatterResidual(connPtr, conn, elementInd, localRes, residual);
    }
  }
}

#ifdef __CUDACC__
template <int order, int numStates, int numDim, A2D::GreenStrainType strainType = A2D::GreenStrainType::LINEAR>
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

  // One element per thread
  const int elementInd = blockIdx.x * blockDim.x + threadIdx.x;
  if (elementInd < numElements) {
    // printf("Running `assemblePlaneStressResidualKernel` element %d\n", elementInd);
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
        interpRealGradient<order, numDim, numDim>(quadPtXi, localNodeStates, JInv, dudx);

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

  // One element per thread
  const int elementInd = blockIdx.x * blockDim.x + threadIdx.x;
  if (elementInd < numElements) {
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
        interpRealGradient<order, numDim, numDim>(quadPtXi, localNodeStates, JInv, dudx);

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
            lagrangePoly2dDeriv<double, order>(quadPtXi, nodeXInd, nodeYInd, N, dudxiDot.get_data());
            // Now propogate through dudx = dudxi * J^-1
            A2D::MatMatMult(dudxiDot, JInv, dudxDot);
            for (int stateInd = 0; stateInd < numStates; stateInd++) {
              // Now we will do a forward AD pass through the weak residual calculation by setting dudxDot as the seed
              // in the state gradient
              for (int ii = 0; ii < numStates; ii++) {
                for (int jj = 0; jj < numDim; jj++) {
                  dudxFwd(ii, jj).deriv[0] = 0.0;
                }
              }
              for (int jj = 0; jj < numDim; jj++) {
                dudxFwd(stateInd, jj).deriv[0] = dudxDot[jj];
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
#ifndef NDEBUG
              printf("QuadPt (%d, %d) contribution to column %d = ", quadPtXInd, quadPtYInd, colInd);
              printMat(matColContribution);
#endif
              for (int ii = 0; ii < numNodes; ii++) {
                for (int jj = 0; jj < numStates; jj++) {
                  const int rowInd = ii * numStates + jj;
                  localMat(rowInd, colInd) += matColContribution(ii, jj);
                }
              }
            }
          }
        }
      }
    }
#ifndef NDEBUG
    printf("Local element Jacobian = ");
    printMat(localMat);
#endif
    // --- Scatter the local element Jacobian into the global matrix ---
    scatterMat<numNodes, numStates>(connPtr,
                                    conn,
                                    elementInd,
                                    &bcsrMap[elementInd * numNodes * numNodes],
                                    localMat,
                                    matEntries);

    // --- Scatter local residual back to global array---
    if (residual != nullptr) {
      scatterResidual(connPtr, conn, elementInd, localRes, residual);
    }
  }
}
#endif

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
  const int threadsPerBlock = 4 * 32;
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
#endif
  auto t1 = std::chrono::high_resolution_clock::now();

#ifdef __CUDACC__
#define ASSEMBLE_PLANE_STRESS_RESIDUAL(elementOrder)                                                                   \
  assemblePlaneStressResidualKernel<elementOrder, 2, 2>                                                                \
      <<<numBlocks, threadsPerBlock>>>(connPtr, conn, numElements, states, nodeCoords, E, nu, t, residual);
#else
#define ASSEMBLE_PLANE_STRESS_RESIDUAL(elementOrder)                                                                   \
  assemblePlaneStressResidual<elementOrder, 2, 2>(connPtr, conn, numElements, states, nodeCoords, E, nu, t, residual);
#endif

  switch (elementOrder) {
    case 1:
      ASSEMBLE_PLANE_STRESS_RESIDUAL(1);
      break;
    case 2:
      ASSEMBLE_PLANE_STRESS_RESIDUAL(2);
      break;
    case 3:
      ASSEMBLE_PLANE_STRESS_RESIDUAL(3);
      break;
    case 4:
      ASSEMBLE_PLANE_STRESS_RESIDUAL(4);
      break;
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
  const int threadsPerBlock = 4 * 32;
  const int numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
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
  assemblePlaneStressJacobianKernel<elementOrder, 2, 2><<<numBlocks, threadsPerBlock>>>(connPtr,                       \
                                                                                        conn,                          \
                                                                                        numElements,                   \
                                                                                        states,                        \
                                                                                        nodeCoords,                    \
                                                                                        E,                             \
                                                                                        nu,                            \
                                                                                        t,                             \
                                                                                        elementBCSRMap,                \
                                                                                        residual,                      \
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
    case 1:
      ASSEMBLE_PLANE_STRESS_JACOBIAN(1);
      break;
    case 2:
      ASSEMBLE_PLANE_STRESS_JACOBIAN(2);
      break;
    case 3:
      ASSEMBLE_PLANE_STRESS_JACOBIAN(3);
      break;
    case 4:
      ASSEMBLE_PLANE_STRESS_JACOBIAN(4);
      break;
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
