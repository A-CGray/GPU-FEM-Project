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

// =============================================================================
// Extension Includes
// =============================================================================
// #include "../element/ElementStruct.h"
#include "../element/ResidualKernelPrototype.h"
#include "TACSAssembler.h"
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

    // --- Write out solution file with analytic displacement field ---
    setAnalyticDisplacements(assembler, displacementField);
    writeTACSSolution(assembler, "output.f5");

    // --- Evaluate Jacobian and residual and write them to file ---
    TACSSchurMat *mat = assembler->createSchurMat(nodeOrdering);
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
    auto t1 = std::chrono::high_resolution_clock::now();
    assembler->assembleRes(res);
    auto t2 = std::chrono::high_resolution_clock::now();
    /* Getting number of seconds as a double. */
    std::chrono::duration<double> tmp = t2 - t1;
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

    int *d_numElements;
    cudaMalloc(&d_numElements, sizeof(int));
    cudaMemcpy(d_numElements, numElements, sizeof(int), cudaMemcpyHostToDevice);

    double *d_disp;
    cudaMalloc(&d_disp, 2 * numNodes * sizeof(double));
    cudaMemcpy(d_disp, disp, 2 * numNodes * sizeof(double));

    double *d_xPts2d;
    cudaMalloc(&d_xPts2d, 2 * numNodes * sizeof(double));
    cudaMemcpy(d_xPts2d, xPts2d, 2 * numNodes * sizeof(double));

    double *d_quadPtWeights;
    cudaMalloc(&d_quadPtWeights, numQuadPts * sizeof(double));
    cudaMemcpy(d_quadPtWeights, quadPtWeights, numQuadPts * sizeof(double));

    double *d_quadPointdNdxi;
    cudaMalloc(&d_quadPointdNdxi, numQuadPts * numNodesPerElement * 2 * sizeof(double));
    cudaMemcpy(d_quadPointdNdxi, quadPointdNdxi, numQuadPts * numNodesPerElement * 2 * sizeof(double));

    double *d_E;
    cudaMalloc(&d_E, sizeof(double));
    cudaMemcpy(d_E, &E, sizeof(double));

    double *d_nu;
    cudaMalloc(&d_nu, sizeof(double));
    cudaMemcpy(d_nu, &nu, sizeof(double));

    double *d_t;
    cudaMalloc(&d_t, sizeof(double));
    cudaMemcpy(d_t, &t, sizeof(double));

    double *d_kernelRes;
    cudaMalloc(&d_kernelRes, numNodes * 2 * sizeof(double));
    cudaMemcpy(d_kernelRes, kernelRes, numNodes * 2 * sizeof(double));

    // Figure out how many blocks and threads to use
    threadsPerBlock = 4 * 32;
    numBlocks = (numElements + threadsPerBlock - 1) / threadsPerBlock;
#endif

#ifdef __CUDACC__
    // --- Create timing events ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#else
    t1 = std::chrono::high_resolution_clock::now();
#endif
    // We need a bunch of if statements here because the kernel is templated on the number of nodes, which we only
    // know at runtime
    switch (numNodesPerElement) {
      case 4:
#ifdef __CUDACC__
        assemblePlaneStressResidualKernel<4, 2, 4, 2><<<numBlocks, threadsPerBlock>>>(d_connPtr,
                                                                                      d_conn,
                                                                                      d_numElements,
                                                                                      d_disp,
                                                                                      d_xPts2d,
                                                                                      d_quadPtWeights,
                                                                                      d_quadPointdNdxi,
                                                                                      d_E,
                                                                                      d_nu,
                                                                                      d_t,
                                                                                      d_kernelRes);
#else
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
#endif
        break;
      case 9:
#ifdef __CUDACC__
        assemblePlaneStressResidualKernel<9, 2, 9, 2><<<numBlocks, threadsPerBlock>>>(d_connPtr,
                                                                                      d_conn,
                                                                                      d_numElements,
                                                                                      d_disp,
                                                                                      d_xPts2d,
                                                                                      d_quadPtWeights,
                                                                                      d_quadPointdNdxi,
                                                                                      d_E,
                                                                                      d_nu,
                                                                                      d_t,
                                                                                      d_kernelRes);
#else
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
#endif
        break;
      case 16:
#ifdef __CUDACC__
        assemblePlaneStressResidualKernel<16, 2, 16, 2><<<numBlocks, threadsPerBlock>>>(d_connPtr,
                                                                                        d_conn,
                                                                                        d_numElements,
                                                                                        d_disp,
                                                                                        d_xPts2d,
                                                                                        d_quadPtWeights,
                                                                                        d_quadPointdNdxi,
                                                                                        d_E,
                                                                                        d_nu,
                                                                                        d_t,
                                                                                        d_kernelRes);
#else
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
#endif
        break;
      case 25:
#ifdef __CUDACC__
        assemblePlaneStressResidualKernel<25, 2, 25, 2><<<numBlocks, threadsPerBlock>>>(d_connPtr,
                                                                                        d_conn,
                                                                                        d_numElements,
                                                                                        d_disp,
                                                                                        d_xPts2d,
                                                                                        d_quadPtWeights,
                                                                                        d_quadPointdNdxi,
                                                                                        d_E,
                                                                                        d_nu,
                                                                                        d_t,
                                                                                        d_kernelRes);
#else
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
#endif
        break;
      default:
        break;
    }
#ifdef __CUDACC__
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float runTime;
    cudaEventElapsedTime(&runTime, start, stop);
    runTime /= 1000; // Convert to seconds
    const double kernelResTime = double(runTime);

    // Copy the residual back to the CPU
    cudaMemcpy(kernelRes, d_kernelRes, numNodes * 2 * sizeof(double), cudaMemcpyDeviceToHost);
#else
    t2 = std::chrono::high_resolution_clock::now();
    /* Getting number of seconds as a double. */
    tmp = t2 - t1;
    const double kernelResTime = tmp.count();
#endif

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
#ifdef __CUDACC__
    cudaFree(d_connPtr);
    cudaFree(d_conn);
    cudaFree(d_numElements);
    cudaFree(d_disp);
    cudaFree(d_xPts2d);
    cudaFree(d_quadPtWeights);
    cudaFree(d_quadPointdNdxi);
    cudaFree(d_E);
    cudaFree(d_nu);
    cudaFree(d_t);
    cudaFree(d_kernelRes);
#endif
  }

  MPI_Finalize();
  return 0;
}
