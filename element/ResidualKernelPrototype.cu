#include "ResidualKernelPrototype.cuh"

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
                                       double *const residual) {
  const int globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (globalThreadIdx < numElements) {
    // Get the element nodal states and coordinates
    double localNodeCoords[numNodes * numDim];
    double localStates[numNodes * numStates];
    for (int ii = 0; ii < numNodes; ii++) {
      for (int jj = 0; jj < numDim; jj++) {
        localNodeCoords[ii * numDim + jj] = nodeCoords[connectivity[globalThreadIdx * numNodes + ii] * numDim + jj];
      }
      for (int jj = 0; jj < numStates; jj++) {
        localStates[ii * numStates + jj] = states[connectivity[globalThreadIdx * numNodes + ii] * numStates + jj];
      }
    }

    double localRes[numNodes * numStates];
    for (int ii = 0; ii < numNodes * numStates; ii++) {
      localRes[ii] = 0.0;
    }

    // Now the main quadrature loop
    for (int quadPtInd = 0; quadPtInd < numQuadPts; quadPtInd++) {
      // Compute Jacobian
      A2D::Mat<double, numDim, numDim> J, JInv;
      const double *const dNdxi = &quadPointdNdxi[quadPtInd * numNodes * numDim];
      interpParamGradient<numNodes, numDim, numDim>(localNodeCoords, dNdxi, J);

      // --- Compute J^-1 and detJ ---
      A2D::MatInv(J, JInv);
      const double detJ = A2D::MatDet(J);

      // --- Compute state gradient in physical space ---
      double dudx[numStates * numDim];
      interpRealGradient(localNodeStates, dNdxi, dudx);

      // Compute weak residual integrand (derivative of energy w.r.t state gradient scaled by quadrature weight and
      // detJ)
      double weakRes[numDim * numDim];
      planeStressWeakRes(dudx, E, nu, quadPtWeights[quadPtInd] * detJ weakRes);

      // Add to residual (transform sensitivity to be w.r.t nodal states and scale by quadrature weight and detJ)
      addTransformStateGradSens<numNodes, numStates, numDim>(weakRes, JInv, dNdxi, localRes);
    }
    // --- Scatter local residual back to global array, need to use an atomic add ---
    for (int ii = 0; ii < numNodes; ii++) {
      const int globalNodeInd = connectivity[globalThreadIdx * numNodes + ii];
      for (int jj = 0; jj < numStates; jj++) {
        atomicAdd(&residual[globalNodeInd * numStates + jj], localRes[ii * numStates + jj]);
      }
    }
  }
}

template <int numNodes, int numVals, int numDim>
__DEVICE__ void addTransformStateGradSens(const double stateGradSens[numVals * numDim],
                                          const double Jinv[numDim * numDim],
                                          const double dNdxi[numNodes * numDim],
                                          double nodalValSens[]) {
  // dfdq = NPrimeParam * J^-1 * dfduPrime^T

  // Number of ops to compute (NPrimeParam * J^-1) * dfduPrime^T
  constexpr int cost1 = numNodes * numDim * numDim + numDim * numDim * numVals;
  // Number of ops to compute NPrimeParam * (J^-1 * dfduPrime^T)
  constexpr int cost2 = numDim * numDim * numVals + numNodes * numDim * numVals;

  if constexpr (cost1 > cost2) {
    // --- Compute J^-1 * dfduPrime^T ---
    double temp[numDim * numVals];
    for (int ii = 0; ii < numDim; ii++) {
      for (int jj = 0; jj < numVals; jj++) {
        temp[ii, jj] = 0.0;
        for (int kk = 0; kk < numDim; kk++) {
          temp[ii, jj] += Jinv[ii * numDim + kk] * stateGradSens[jj * numVals + kk];
        }
      }
    }
    // --- Add NPrimeParam * temp ---
    for (int ii = 0; ii < numNodes; ii++) {
      for (int jj = 0; jj < numVals; jj++) {
        nodalValSens[ii * numVals + jj] = 0.0;
        for (int kk = 0; kk < numDim; kk++) {
          nodalValSens[ii * numVals + jj] += dNdxi[ii * numDim + kk] * temp[kk * numVals + jj];
        }
      }
    }
  }
  else {
    // --- Compute (NPrimeParam * J^-1) ---
    double temp[numNodes * numDim];
    for (int ii = 0; ii < numNodes; ii++) {
      for (int jj = 0; jj < numDim; jj++) {
        temp[ii, jj] = 0.0;
        for (int kk = 0; kk < numDim; kk++) {
          temp[ii, jj] += dNdxi[ii * numDim + kk] * Jinv[kk * numDim + jj];
        }
      }
    }
    // --- Add temp * dfduPrime^T ---
    for (int ii = 0; ii < numNodes; ii++) {
      for (int jj = 0; jj < numVals; jj++) {
        for (int kk = 0; kk < numDim; kk++) {
          nodalValSens[ii * numVals + jj] += temp[ii * numDim + kk] * stateGradSens[jj * numVals + kk];
        }
      }
    }
  }
}

// dudxi = localNodeValues^T * dNdxi
template <int numNodes, int numVals, int numDim>
__DEVICE__ void interpParamGradient(const double nodalValues[numNodes * numVals],
                                    const double dNdxi[numNodes * numDim],
                                    double dudxi[numVals * numDim]) {
  for (int ii = 0; ii < numVals; ii++) {
    for (int jj = 0; jj < numDim; jj++) {
      dudxi[ii, jj] = 0.;
      for (int kk = 0; kk < numNodes; kk++) {
        dudxi[ii * numDim + jj] += nodalValues[kk * numVals + ii] * dNdxi[kk * numDim + jj]
      }
    }
  }
}

// dudx = localNodeValues^T * dNdxi * Jinv
template <int numNodes, int numVals, int numDim>
__DEVICE__ void interpRealGradient(const double nodalValues[numNodes * numVals],
                                   const double dNdxi[numNodes * numDim],
                                   const double Jinv[numDim * numDim],
                                   double dudx[numVals * numDim]) {
  // dudxi = localNodeStates^T * dNdxi
  // dudx = dudxi * J^-1 = localNodeStates^T * dNdxi * J^-1
  // most efficient (in terms of FLOPS) to compute dudxi = localNodeStates^T * dNdxi first, then dudx = dudxi *
  // J^-1

  // dudxi = localNodeValues ^ T * dNdxi
  double dudxi[numVals * numDim];
  interpParamGradient<numNodes, numVals, numDim>(nodalValues, dNdxi, dudxi);

  // dudx = dudxi * J^-1
  for (int ii = 0; ii < numVals; ii++) {
    for (int jj = 0; jj < numDim) {
      dudx[ii * numDim + jj] = 0.0;
      for (int kk = 0; kk < numDim; kk++) {
        dudx[ii * numDim + jj] += dudxi[ii * numDim + kk] * Jinv[kk * numDim + jj];
      }
    }
  }
}

template <GreenStrainType strainType = GreenStrainType::LINEAR>
__DEVICE__ void
planeStressWeakRes(const double uPrime[4], const double E, const double nu, const double scale, double residual[4]) {
  A2D::ADObj<A2D::A2DMat<double, 2, 2>> uPrimeMat(uPrime);
  A2D::ADObj<A2D::SymMat<double, N>> strain, stress;
  A2D::ADObj<double> energy;

  const double mu = 0.5 * E / (1.0 + nu);
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  auto stack = A2D::MakeStack(A2D::MatGreenStrain<strainType>(uPrimeMat, strain),
                              SymIsotropic(mu, lambda, strain, stress),
                              SymMatMultTrace(strain, stress, energy));
  // Compute strain energy derivative w.r.t state gradient
  energy.bvalue() = 0.5 * scale; // Set the seed value (0.5 because energy = 0.5 * sigma : epsilon)
  stack.reverse();               // Reverse mode AD through the stack
  for (int ii = 0; ii < 4; ii++) {
    residual[ii] = uPrimeMat.bvalue().data[ii];
  }
}
