import numpy as np


def interpGradient(nodalValues, dNdxi, dudxi):
    numVals = nodalValues.shape[1]
    numNodes = nodalValues.shape[0]
    numDim = dNdxi.shape[1]
    for ii in range(numVals):
        for jj in range(numDim):
            dudxi[ii, jj] = 0.0
            for kk in range(numNodes):
                dudxi[ii, jj] += nodalValues.flatten()[kk * numVals + ii] * dNdxi.flatten()[kk * numDim + jj]


numNodes = 4
numDim = 2
numStates = 2

q = np.random.rand(numNodes, numStates)
dndxi = np.random.rand(numNodes, numDim)
Jinv = np.random.rand(numDim, numDim)
dEduPrime = np.random.rand(numStates, numDim)
einsumString = "sn,nd,dD->sD"
path, result = np.einsum_path(einsumString, q.T, dndxi, Jinv, optimize="optimal")
print(result)

einsumString = "nd,dD,Ds->ns"
path, result = np.einsum_path(einsumString, dndxi, Jinv, dEduPrime.T, optimize="optimal")
print(result)

# dudxi = np.empty((numStates, numDim))
# interpGradient(q, dndxi, dudxi)
# print(f"{dudxi=}")
