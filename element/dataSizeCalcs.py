import numpy as np

numDim = 2
numStates = 2

threadRegisterLimit = 256
totalRegisterLimit = 65536
sharedMemLimit = 32768
blockThreadLimit = 1024
doubleSize = 8

for elementOrder in range(1, 5):

    elemintLimitNumThreads = int(np.floor(blockThreadLimit / (elementOrder + 1) ** numDim))
    print(f"\nElement order {elementOrder}:")
    print(f"Max elements per block to stay under thread limit = {elemintLimitNumThreads}")

    numNodes = (elementOrder + 1) ** numDim
    localStateSize = numNodes * numStates * doubleSize
    localCoordSize = numNodes * numDim * doubleSize
    localResSize = localStateSize
    sharedDataSize = localStateSize + localCoordSize + localResSize
    elementLimitSharedMem = int(np.floor(sharedMemLimit / (sharedDataSize)))
    print(f"Local element data (shared mem) = {sharedDataSize} bytes")
    print(f"Max elements per block to stay under shared mem limit = {elementLimitSharedMem}")

    # Data at each quadrature point
    quadPointWeights = numDim + 1
    quadPtCoords = numDim
    Jacs = numDim * numDim * 2  # jacobian and it's inverse
    stateGradients = (
        numStates * numDim * 3
    )  # We have the state gradient then the copy used inside the weak res calculation and it's reverse seed
    weakRes = stateGradients
    basisGradient = numDim  # Will only compute one basis gradient at a time
    stresses = 3 * 2  # 2x2 symmetric stress tensor and it's reverse seed
    strains = stresses
    energy = 2  # strain energy and it's reverse seed
    quadPtDataSize = doubleSize * (
        quadPointWeights + quadPtCoords + Jacs + stateGradients + weakRes + basisGradient + stresses + strains + energy
    )
    print(f"Data at each quadrature point = {quadPtDataSize} bytes")
    elementLimitRegisters = int(np.floor(totalRegisterLimit / min(quadPtDataSize, threadRegisterLimit)))
    print(f"Max elements per block to stay under register limit = {elementLimitRegisters}")
