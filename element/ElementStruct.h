/*
=============================================================================

=============================================================================
@File    :   ElementStruct.h
@Date    :   2024/11/11
@Author  :   Alasdair Christison Gray
@Description :
*/

// =============================================================================
// Standard Library Includes
// =============================================================================

// =============================================================================
// Extension Includes
// =============================================================================
#include "a2dcore.h"

// =============================================================================
// Global constant definitions
// =============================================================================
// TODO: How do we want to implement this?
// Option 1: Use a struct, pass struct to kernel
// - Pros: only includes data, don't need to worry about how to implement a kernel within a class
// - Cons:
// - need to copy struct to GPU, if kernel contains pointers then this is complicated as need to do a deep copy
// where you copy the data pointed to by the pointers and then update the pointers in the struct to point to the new,
// could do this automatically with CUDA managed memory
// Option 2: Use a class, implement kernel as a method
struct elementParams {
    // Numbers of things
    const int numNodes;
    const int numParamDim;
    const int numDim;
    const int numStates;
    const int numIntPoints;
    // Values of things
    const double *const intPointWeights;
    const double *const intPointN;
    const double *const intPointNPrime;
};

template <typename numType, int numNodes, int numStates, int numDims>
__device__ void planeStressResQuadPt(const numType *const nodeCoords,
                                     const numType *const nodeStates,
                                     const numType *const NPrimeParam,
                                     const numType E,
                                     const numType nu,
                                     numType res[]) {
  ADObj<A2DMat<numType, numNodes, numStates>> q(nodeStates);
  A2DMat<numType, numNodes, numDims> xNodes(nodeCoords);
  A2DMat<numType, numDims, numDims> J, Jinv;
}
