numDim = 2
numVals = 2
for numNodes in [4, 9, 16, 25]:
    cost1 = numNodes * numDim**2 + numDim**2 * numVals
    cost2 = numDim**2 * numVals + numNodes * numDim * numVals
    cost3 = numNodes * numVals * numDim**2
    print(f"\n{numNodes=}:")
    print(f"Cost1 = {cost1}")
    print(f"Cost2 = {cost2}")
    print(f"Cost3 = {cost3}")
    print(f"Min cost = {min(cost1, cost2, cost3)}")
