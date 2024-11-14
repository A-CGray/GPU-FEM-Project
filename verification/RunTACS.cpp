/*
=============================================================================

=============================================================================
@File    :   RunTACS.cpp
@Date    :   2024/11/07
@Author  :   Alasdair Christison Gray
@Description :
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
// #include "../element/ElementStruct.h"
#include "TACSHelpers.h"

// =============================================================================
// Global constant definitions
// =============================================================================

// =============================================================================
// Function prototypes
// =============================================================================

// =============================================================================
// Main
// =============================================================================
int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  // The TACSAssembler object - which should be allocated if the mesh
  // is loaded correctly
  TACSAssembler *assembler = NULL;

  // Try to load the input file as a BDF file through the
  // TACSMeshLoader class
  if (argc > 1) {
    createTACSAssembler(argv[1], assembler);
  }
  else {
    fprintf(stderr, "No BDF file provided\n");
  }

  if (assembler) {

    // --- Write out solution file with analytic displacement field ---
    setAnalyticDisplacements(assembler, displacementField);
    writeTACSSolution(assembler, "output.f5");

    // --- Evaluate Jacobian and residual and write them to file ---
    TACSAssembler::OrderingType matOrder = TACSAssembler::AMD_ORDER;
    TACSSchurMat *mat = assembler->createSchurMat(matOrder);
    assembler->assembleJacobian(1.0, 0.0, 0.0, NULL, mat);
    BCSRMat *jac;
    mat->getBCSRMat(&jac, NULL, NULL, NULL);
    BCSRMatData *jacData = jac->getMatData();
    printf("Jacobian data:\n");
    printf("nRows: %d\n", jacData->nrows);
    printf("nCols: %d\n", jacData->ncols);
    printf("Block size: %d\n", jacData->bsize);
    writeBCSRMatToFile(jacData, "TACSJacobian.mtx");

    // --- Evaluate residual and write to file ---
    // TODO: Why does the ordering of the residual not seem to be affected by the matrix ordering type?
    TACSBVec *res = assembler->createVec();
    assembler->assembleRes(res);
    assembler->reorderVec(res);
    if (assembler->isReordered()) {
      printf("Assembler is reordered\n");
    }
    else {
      printf("Assembler is not reordered\n");
    }
    writeResidualToFile(res, "TACSResidual.csv");

    // --- Get the data required for the GPU kernel ---
    // number of nodes & elements
    // Node coordinates
    // Element Connectivity
    // Integration point weights
    // Basis function gradients at the integration points
    const int numNodes = assembler->getNumNodes();
    const int numElements = assembler->getNumElements();
    const int *connPtr;
    const int *conn;
    assembler->getElementConnectivity(&connPtr, &conn);

    // Create a node vector
    TACSBVec *nodeVec = assembler->createNodeVec();
    nodeVec->incref();
    assembler->getNodes(nodeVec);

    // Get the local node locations array
    TacsScalar *Xpts = NULL;
    nodeVec->getArray(&Xpts);

    // Assume all elements are the same so we can just use the first one
    TACSElement *const element = assembler->getElements()[0];
    TACSElementBasis *const basis = element->getElementBasis();
    const int numIntPoints = element->getNumQuadraturePoints();
    const int numNodesPerElement = element->getNumNodes();
    double *const intPointWeights = new double[numIntPoints];
    double *const intPointN = new double[numIntPoints * numNodesPerElement];
    // intPointNPrimeParam is a numIntPoints x numNodesPerElement x 2 array
    // intPointNPrimeParam[i, j, k] is the kth component of the gradient (in parametric space) of the jth basis
    // function at the ith integration point (dN_j/dx_k at gauss point i)
    double *const intPointNPrimeParam = new double[numIntPoints * numNodesPerElement * 2];

    for (int ii = 0; ii < numIntPoints; ii++) {
      double pt[3];
      intPointWeights[ii] = basis->getQuadraturePoint(ii, pt);
      basis->computeBasisGradient(pt,
                                  &intPointN[ii * numNodesPerElement],
                                  &intPointNPrimeParam[ii * numNodesPerElement * 2]);
      printf("\nInt point %d:\nweight = %f\nN = [", ii, intPointWeights[ii]);
      for (int jj = 0; jj < numNodesPerElement; jj++) {
        printf("%f ", intPointN[ii * numNodesPerElement + jj]);
      }
      printf("]\ndNdxi = [");
      for (int jj = 0; jj < numNodesPerElement; jj++) {
        printf("%f, %f\n",
               intPointNPrimeParam[ii * (numNodesPerElement * 2) + (2 * jj)],
               intPointNPrimeParam[ii * (numNodesPerElement * 2) + (2 * jj) + 1]);
      }
      printf("]\n");
    }

    // Free memory
    delete[] intPointWeights;
    delete[] intPointN;
    delete[] intPointNPrimeParam;
  }

  MPI_Finalize();
  return 0;
}

// =============================================================================
// Function definitions
// =============================================================================

// template <elementParams elemParams>
// void a2dResidual(const int numElements,
//                  const int *const connPtr,
//                  const int *const conn,
//                  const TacsScalar *const Xpts,
//                  const TacsScalar *const states,
//                  const TacsScalar E,
//                  const TacsScalar nu,
//                  const TacsScalar t,
//                  TacsScalar *const res) {
//   // // Create the energy stack which we will reuse for each element
//   // // Inputs:
//   // Mat<TacsScalar, elemParams.numNodes, elemParams.numDim> nodeCoords;
//   // Mat<TacsScalar, elemParams.numNodes, elemParams.nunDim> NPrimeParam;
//   // ADObj<Mat<TacsScalar, elemParams.numNodes, elemParams.numStates>> nodeStates;

//   // // Intermediate variables
//   // ADObj<Mat<TacsScalar, elemParams.numDim, elemParams.numDim>> J, Jinv;
//   // ADObj<Mat<TacsScalar, elemParams.numStates, elemParams.numDim>> uPrime, uPrimeParam, F;
//   // ADObj<Mat<TacsScalar, 2, 2>> stress, strain;
//   // ADObj<TacsScalar> Energy;

//   // auto EnergyStack = MakeStack(
//   //     // J = NPrimeParam^T * nodeCoords
//   //     // Jinv = inv(J)
//   //     // uPrimeParam = NPrimeParam^T * nodeStates
//   //     // uPrime = (Jinv * uPrimeParam)^T
//   //     // F = uPrime + I
//   //     // strain = 0.5 * (F^T * F - I) or 0.5 * (F + F^T) - I
//   //     // Stress = 2*mu*strain + lambda*tr(strain)*I
//   //     // Energy = 0.5 * tr(strain * stress)
//   // );

//   for (int ii = 0; ii < numElements; ii++) {
//   }
// }
