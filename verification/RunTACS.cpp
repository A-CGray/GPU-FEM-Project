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
#include <chrono>
#include <math.h>

// =============================================================================
// Extension Includes
// =============================================================================
// #include "../element/ElementStruct.h"
#include "../element/FEKernels.h"
#include "TACSAssembler.h"
#include "TACSElementVerification.h"
#include "TACSHelpers.h"

// =============================================================================
// Global constant definitions
// =============================================================================
const int MAX_TIMING_LOOPS = 100;
const double MAX_TIME = 30;
const int JAC_WRITE_SIZE_LIMIT = 1000;
const int RES_WRITE_SIZE_LIMIT = 1000;

// ==============================================================================
// Helper functions
// ==============================================================================
void computeTimingStats(const double runTimes[], const int numLoopsRun) {

  double minTime = runTimes[0];
  double maxTime = runTimes[0];
  double avgTime = runTimes[0];
  for (int ii = 1; ii < numLoopsRun; ii++) {
    minTime = std::min(minTime, runTimes[ii]);
    maxTime = std::max(maxTime, runTimes[ii]);
    avgTime += runTimes[ii];
  }
  avgTime /= numLoopsRun;
  double stdDevTime = 0;
  for (int ii = 0; ii < numLoopsRun; ii++) {
    stdDevTime += (runTimes[ii] - avgTime) * (runTimes[ii] - avgTime);
  }
  stdDevTime = sqrt(stdDevTime / numLoopsRun);

  // Print out the time taken for each part of the computation
  printf("Timing statistics over %d runs:\n", numLoopsRun);
  printf("    Min = %.11e\n", minTime);
  printf("    Max = %.11e\n", maxTime);
  printf("    Avg = %.11e\n", avgTime);
  printf("Std Dev = %.11e\n", stdDevTime);
}

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

  // TODO: Figure out why TACS and kernel residual order doesn't match if I choose non-natural ordering
  TACSAssembler::OrderingType nodeOrdering = TACSAssembler::NATURAL_ORDER;

  // Try to load the input file as a BDF file through the
  // TACSMeshLoader class
  if (argc > 1) {
    setupTACS(argv[1], nodeOrdering, assembler, mesh, props, stiff, model, basis, element);
  }
  else {
    fprintf(stderr, "No BDF file provided\n");
  }

  if (assembler) {
    const int numNodes = assembler->getNumNodes();
    const int numElements = assembler->getNumElements();
    const int numQuadPts = element->getNumQuadraturePoints();
    const int numNodesPerElement = element->getNumNodes();
    const int *connPtr;
    const int *conn;
    assembler->getElementConnectivity(&connPtr, &conn);

    // --- Write out solution file with analytic displacement field ---
    setAnalyticDisplacements(assembler, displacementField);
    writeTACSSolution(assembler, "output.f5");

    // --- Evaluate Jacobian and write out up to 1000 rows ---
    TACSSchurMat *mat = assembler->createSchurMat(nodeOrdering);

    auto t1 = std::chrono::high_resolution_clock::now();
    assembler->assembleJacobian(1.0, 0.0, 0.0, NULL, mat);
    auto t2 = std::chrono::high_resolution_clock::now();
    /* Getting number of seconds as a double. */
    std::chrono::duration<double> tmp = t2 - t1;
    const double tacsJacTime = tmp.count();
    BCSRMat *jac;
    mat->getBCSRMat(&jac, NULL, NULL, NULL);
    BCSRMatData *jacData = jac->getMatData();
    printf("Jacobian data:\n");
    printf("nRows: %d\n", jacData->nrows);
    printf("nCols: %d\n", jacData->ncols);
    printf("Block size: %d\n", jacData->bsize);

    writeBCSRMatToFile(jacData, "TACSJacobian.mtx", JAC_WRITE_SIZE_LIMIT / 2, JAC_WRITE_SIZE_LIMIT / 2);
    mat->zeroEntries();

    int *elementBCSRMap;
    const int elementBCSRMapSize = numElements * numNodesPerElement * numNodesPerElement;
    elementBCSRMap = new int[elementBCSRMapSize];
    generateElementBCSRMap(connPtr, conn, numElements, numNodesPerElement, jacData, elementBCSRMap);

    // --- Evaluate residual and write to file ---
    // TODO: Why does the ordering of the residual not seem to be affected by the matrix ordering type?
    TACSBVec *res = assembler->createVec();
    t1 = std::chrono::high_resolution_clock::now();
    assembler->assembleRes(res);
    t2 = std::chrono::high_resolution_clock::now();
    /* Getting number of seconds as a double. */
    tmp = t2 - t1;
    const double tacsResTime = tmp.count();
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
    // --- Write out the first 1000 entries of the residual ---
    writeArrayToFile<TacsScalar>(tacsResArray, std::min(resSize, RES_WRITE_SIZE_LIMIT), "TACSResidual.csv");

    // --- Get the data required for the kernel ---
    // Material properties
    // Node coordinates and states
    // Element Connectivity
    // Integration point weights
    // Basis function gradients at the integration points

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
    }

    // Create an array for the kernel computed residual then call it
    double *const kernelRes = new double[numNodes * 2];
    memset(kernelRes, 0, numNodes * 2 * sizeof(double));

// If running on the GPU, allocate memory for the GPU data and transfer it over
#ifdef __CUDACC__
    int *d_connPtr;
    cudaMalloc(&d_connPtr, numElements * sizeof(int));
    cudaMemcpy(d_connPtr, connPtr, numElements * sizeof(int), cudaMemcpyHostToDevice);

    int *d_conn;
    cudaMalloc(&d_conn, numElements * numNodesPerElement * sizeof(int));
    cudaMemcpy(d_conn, conn, numElements * numNodesPerElement * sizeof(int), cudaMemcpyHostToDevice);

    double *d_disp;
    cudaMalloc(&d_disp, 2 * numNodes * sizeof(double));
    cudaMemcpy(d_disp, disp, 2 * numNodes * sizeof(double), cudaMemcpyHostToDevice);

    double *d_xPts2d;
    cudaMalloc(&d_xPts2d, 2 * numNodes * sizeof(double));
    cudaMemcpy(d_xPts2d, xPts2d, 2 * numNodes * sizeof(double), cudaMemcpyHostToDevice);

    double *d_quadPtWeights;
    cudaMalloc(&d_quadPtWeights, numQuadPts * sizeof(double));
    cudaMemcpy(d_quadPtWeights, quadPtWeights, numQuadPts * sizeof(double), cudaMemcpyHostToDevice);

    double *d_quadPointdNdxi;
    cudaMalloc(&d_quadPointdNdxi, numQuadPts * numNodesPerElement * 2 * sizeof(double));
    cudaMemcpy(d_quadPointdNdxi,
               quadPointdNdxi,
               numQuadPts * numNodesPerElement * 2 * sizeof(double),
               cudaMemcpyHostToDevice);

    double *d_kernelRes;
    cudaMalloc(&d_kernelRes, numNodes * 2 * sizeof(double));

    double *d_matEntries;
    int matDataLength = jacData->bsize * jacData->bsize * jacData->rowp[jacData->nrows];
    cudaMalloc(&d_matEntries, matDataLength * sizeof(double));

    int *d_elementBCSRMap;
    cudaMalloc(&d_elementBCSRMap, elementBCSRMapSize * sizeof(int));
    cudaMemcpy(d_elementBCSRMap, elementBCSRMap, elementBCSRMapSize * sizeof(int), cudaMemcpyHostToDevice);

#endif

    // ==============================================================================
    // Run timing loop
    // ==============================================================================
    double residualRunTimes[MAX_TIMING_LOOPS], jacobianRunTimes[MAX_TIMING_LOOPS];
    int numLoopsRun = 0;

    auto timingLoopStart = std::chrono::high_resolution_clock::now();
    for (int timingRun = 0; timingRun < MAX_TIMING_LOOPS; timingRun++) {

#ifdef __CUDACC__
      cudaMemset(d_kernelRes, 0, numNodes * 2 * sizeof(double));
      residualRunTimes[numLoopsRun] = runResidualKernel(numNodesPerElement,
                                                        d_connPtr,
                                                        d_conn,
                                                        numElements,
                                                        d_disp,
                                                        d_xPts2d,
                                                        d_quadPtWeights,
                                                        d_quadPointdNdxi,
                                                        E,
                                                        nu,
                                                        t,
                                                        d_kernelRes);

      cudaMemset(d_matEntries, 0, matDataLength * sizeof(double));
      jacobianRunTimes[numLoopsRun] = runJacobianKernel(numNodesPerElement,
                                                        d_connPtr,
                                                        d_conn,
                                                        numElements,
                                                        d_disp,
                                                        d_xPts2d,
                                                        d_quadPtWeights,
                                                        d_quadPointdNdxi,
                                                        E,
                                                        nu,
                                                        t,
                                                        d_elementBCSRMap,
                                                        nullptr,
                                                        d_matEntries);
#else
      memset(kernelRes, 0, numNodes * 2 * sizeof(double));
      residualRunTimes[numLoopsRun] = runResidualKernel(numNodesPerElement,
                                                        connPtr,
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

      mat->zeroEntries();
      jacobianRunTimes[numLoopsRun] = runJacobianKernel(numNodesPerElement,
                                                        connPtr,
                                                        conn,
                                                        numElements,
                                                        disp,
                                                        xPts2d,
                                                        quadPtWeights,
                                                        quadPointdNdxi,
                                                        E,
                                                        nu,
                                                        t,
                                                        elementBCSRMap,
                                                        nullptr,
                                                        jacData->A);
#endif
      numLoopsRun++;
      std::chrono::duration<double> tmp = std::chrono::high_resolution_clock::now() - timingLoopStart;
      const double totalRunTime = tmp.count();
      printf("Timing run %4d: res time = %.11e s, jac time = %.11e, total time = %f\n",
             numLoopsRun,
             residualRunTimes[numLoopsRun - 1],
             jacobianRunTimes[numLoopsRun - 1],
             totalRunTime);
      if (totalRunTime > MAX_TIME) {
        break;
      }
    }

#ifdef __CUDACC__
    // Copy the residual back to the CPU
    cudaMemcpy(kernelRes, d_kernelRes, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);
    // Copy the mat entries into the TACS matrix
    cudaMemcpy(jacData->A, d_matEntries, matDataLength * sizeof(double), cudaMemcpyDeviceToHost);
#endif

    // --- Write out the first 1000 entries of the residual ---
    writeArrayToFile<TacsScalar>(kernelRes, std::min(resSize, RES_WRITE_SIZE_LIMIT), "KernelResidual.csv");

    // --- Write out the first 1000 rows and columns of the Jacobian ---
    writeBCSRMatToFile(jacData, "KernelJacobian.mtx", JAC_WRITE_SIZE_LIMIT / 2, JAC_WRITE_SIZE_LIMIT / 2);

    // Compute the error in the residual
    int maxAbsErrInd, maxRelErrorInd;
    double maxAbsError = TacsGetMaxError(kernelRes, tacsResArray, resSize, &maxAbsErrInd);
    double maxRelError = TacsGetMaxRelError(kernelRes, tacsResArray, resSize, &maxRelErrorInd);
    bool residualMatch = TacsAssertAllClose(kernelRes, tacsResArray, resSize, 1e-6, 1e-6);
    if (residualMatch) {
      printf("\n\nResiduals match\n");
    }
    else {
      printf("Residuals do not match\n");
    }
    printf("Max Err: %10.4e in component %d.\n", maxAbsError, maxAbsErrInd);
    printf("Max REr: %10.4e in component %d.\n", maxRelError, maxRelErrorInd);

    // Compute timing stats for residual and jacobian kernels
    printf("\n\nTiming stats for problem with %d CQUAD%d elements and %d DOF:\n",
           numElements,
           numNodesPerElement,
           numNodes * 2);
    printf("Residual kernel:\n");
    computeTimingStats(residualRunTimes, numLoopsRun);

    printf("\n\nJacobian kernel:\n");
    computeTimingStats(jacobianRunTimes, numLoopsRun);

    printf("\n\nTime taken for a single TACS residual assembly was %f s\n", tacsResTime);
    printf("Time taken for a single TACS jacobian assembly was %f s\n", tacsJacTime);

    // Write timing results to files
    writeArrayToFile(residualRunTimes, numLoopsRun, "ResidualTimings.csv");
    writeArrayToFile(jacobianRunTimes, numLoopsRun, "JacobianTimings.csv");

    // Free memory
    delete[] quadPtWeights;
    delete[] quadPtN;
    delete[] quadPointdNdxi;
    delete[] xPts2d;
    delete[] kernelRes;
    delete[] elementBCSRMap;
#ifdef __CUDACC__
    cudaFree(d_connPtr);
    cudaFree(d_conn);
    cudaFree(d_disp);
    cudaFree(d_xPts2d);
    cudaFree(d_quadPtWeights);
    cudaFree(d_quadPointdNdxi);
    cudaFree(d_kernelRes);
    cudaFree(d_elementBCSRMap);
#endif
  }

  MPI_Finalize();
  return 0;
}
