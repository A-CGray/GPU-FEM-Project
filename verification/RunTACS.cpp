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
#include <omp.h>

// =============================================================================
// Extension Includes
// =============================================================================
// #include "../element/ElementStruct.h"
#include "../element/ResidualKernelPrototype.h"
#include "TACSElementVerification.h"
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
  TACSMeshLoader *mesh = nullptr;
  TACSMaterialProperties *props = nullptr;
  TACSPlaneStressConstitutive *stiff = nullptr;
  TACSLinearElasticity2D *model = nullptr;
  TACSElementBasis *basis = nullptr;
  TACSElement2D *element = nullptr;

  // Try to load the input file as a BDF file through the
  // TACSMeshLoader class
  if (argc > 1) {
    setupTACS(argv[1], assembler, mesh, props, stiff, model, basis, element);
  }
  else {
    fprintf(stderr, "No BDF file provided\n");
  }

  if (assembler) {

    // --- Write out solution file with analytic displacement field ---
    setAnalyticDisplacements(assembler, displacementField);
    writeTACSSolution(assembler, "output.f5");

    // --- Evaluate Jacobian and residual and write them to file ---
    TACSAssembler::OrderingType matOrder = TACSAssembler::NATURAL_ORDER;
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
    const double tacsResStartTime = omp_get_wtime();
    assembler->assembleRes(res);
    const double tacsResTime = omp_get_wtime() - tacsResStartTime;
    assembler->reorderVec(res);
    if (assembler->isReordered()) {
      printf("Assembler is reordered\n");
    }
    else {
      printf("Assembler is not reordered\n");
    }

    // Get the residual array
    TacsScalar *tacsResArray;
    res->getArray(&tacsResArray);
    int resSize;
    res->getSize(&resSize);
    writeArrayToFile<TacsScalar>(tacsResArray, resSize, "TACSResidual.csv");

    // --- Get the data required for the kernel ---
    // number of nodes & elements
    // Node coordinates and states
    // Element Connectivity
    // Integration point weights
    // Basis function gradients at the integration points
    // Material properties
    const int numNodes = assembler->getNumNodes();
    const int numElements = assembler->getNumElements();
    const int *connPtr;
    const int *conn;
    assembler->getElementConnectivity(&connPtr, &conn);

    // Get the material properties
    TacsScalar E, nu, t;
    props->getIsotropicProperties(&E, &nu);
    stiff->getDesignVars(0, 1, &t);

    // Create a node vector
    TACSBVec *nodeVec = assembler->createNodeVec();
    nodeVec->incref();
    assembler->getNodes(nodeVec);

    // Get the local node coordinates array and convert it to 2D
    TacsScalar *Xpts = NULL;
    nodeVec->getArray(&Xpts);
    double *const xPts2d = new double[numNodes * 2];
    for (int ii = 0; ii < numNodes; ii++) {
      xPts2d[ii * 2] = TacsRealPart(Xpts[ii * 3]);
      xPts2d[ii * 2 + 1] = TacsRealPart(Xpts[ii * 3 + 1]);
    }

    // Get the state variables
    TACSBVec *dispVec = assembler->createVec();
    dispVec->incref();
    assembler->getVariables(dispVec);

    TacsScalar *disp;
    dispVec->getArray(&disp);

    // Get data about the element
    const int numQuadPts = element->getNumQuadraturePoints();
    const int numNodesPerElement = element->getNumNodes();
    double *const quadPtWeights = new double[numQuadPts];
    double *const quadPtN = new double[numQuadPts * numNodesPerElement];
    // quadPointdNdxi is a numQuadPts x numNodesPerElement x 2 array
    // quadPointdNdxi[i, j, k] is the kth component of the gradient (in parametric space) of the jth basis
    // function at the ith integration point (dN_j/dx_k at gauss point i)
    double *const quadPointdNdxi = new double[numQuadPts * numNodesPerElement * 2];

    for (int ii = 0; ii < numQuadPts; ii++) {
      double pt[3];
      quadPtWeights[ii] = basis->getQuadraturePoint(ii, pt);
      basis->computeBasisGradient(pt, &quadPtN[ii * numNodesPerElement], &quadPointdNdxi[ii * numNodesPerElement * 2]);
      printf("\nInt point %d:\nweight = %f\nN = [", ii, quadPtWeights[ii]);
      for (int jj = 0; jj < numNodesPerElement; jj++) {
        printf("%f ", quadPtN[ii * numNodesPerElement + jj]);
      }
      printf("]\ndNdxi = [");
      for (int jj = 0; jj < numNodesPerElement; jj++) {
        printf("%f, %f\n",
               quadPointdNdxi[ii * (numNodesPerElement * 2) + (2 * jj)],
               quadPointdNdxi[ii * (numNodesPerElement * 2) + (2 * jj) + 1]);
      }
      printf("]\n");
    }

    // Create an array for the kernel computed residual then call it
    double *const kernelRes = new double[numNodes * 2];
    memset(kernelRes, 0, numNodes * 2 * sizeof(double));
    // We need a bunch of if statements here because the kernel is templated on the number of nodes, which we only know
    // at runtime
    const double kernelResStartTime = omp_get_wtime();
    switch (numNodesPerElement) {
      case 4:
        assemblePlaneStressResidual<4, 2, 4, 2>(connPtr,
                                                conn,
                                                numElements,
                                                disp,
                                                xPts2d,
                                                quadPtWeights,
                                                quadPointdNdxi,
                                                E,
                                                nu,
                                                t,
                                                kernelRes);
        break;
      case 9:
        assemblePlaneStressResidual<9, 2, 9, 2>(connPtr,
                                                conn,
                                                numElements,
                                                disp,
                                                xPts2d,
                                                quadPtWeights,
                                                quadPointdNdxi,
                                                E,
                                                nu,
                                                t,
                                                kernelRes);
        break;
      case 16:
        assemblePlaneStressResidual<16, 2, 16, 2>(connPtr,
                                                  conn,
                                                  numElements,
                                                  disp,
                                                  xPts2d,
                                                  quadPtWeights,
                                                  quadPointdNdxi,
                                                  E,
                                                  nu,
                                                  t,
                                                  kernelRes);
        break;
      case 25:
        assemblePlaneStressResidual<25, 2, 25, 2>(connPtr,
                                                  conn,
                                                  numElements,
                                                  disp,
                                                  xPts2d,
                                                  quadPtWeights,
                                                  quadPointdNdxi,
                                                  E,
                                                  nu,
                                                  t,
                                                  kernelRes);
        break;
      default:
        break;
    }
    const double kernelResTime = omp_get_wtime() - kernelResStartTime;

    writeArrayToFile<TacsScalar>(kernelRes, resSize, "KernelResidual.csv");

    // Compute the error
    int maxAbsErrInd, maxRelErrorInd;
    double maxAbsError = TacsGetMaxError(kernelRes, tacsResArray, resSize, &maxAbsErrInd);
    double maxRelError = TacsGetMaxRelError(kernelRes, tacsResArray, resSize, &maxRelErrorInd);
    bool residualMatch = TacsAssertAllClose(kernelRes, tacsResArray, resSize, 1e-6, 1e-6);
    if (residualMatch) {
      printf("Residuals match\n");
    }
    else {
      printf("Residuals do not match\n");
    }
    printf("Max Err: %10.4e in component %d.\n", maxAbsError, maxAbsErrInd);
    printf("Max REr: %10.4e in component %d.\n", maxRelError, maxRelErrorInd);

    // Print out the time taken for each part of the computation
    printf("  TACS residual time: %f s\n", tacsResTime);
    printf("Kernel residual time: %f s\n", kernelResTime);

    // Free memory
    delete[] quadPtWeights;
    delete[] quadPtN;
    delete[] quadPointdNdxi;
    delete[] xPts2d;
    delete[] kernelRes;
  }

  MPI_Finalize();
  return 0;
}
