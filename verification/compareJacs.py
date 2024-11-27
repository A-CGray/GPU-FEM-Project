import scipy.sparse as sp


def readSparseCOOMat(fileName: str) -> sp.spmatrix:
    with open(fileName, "r") as f:
        lines = f.readlines()
    data = []
    rows = []
    cols = []
    numRows, numCols, nnz = map(int, lines[0].split())
    for line in lines[1:]:
        l = line.split()

        rows.append(int(l[0]))
        cols.append(int(l[1]))
        data.append(float(l[2]))
    return sp.coo_matrix((data, (rows, cols)), shape=(numRows, numCols))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import niceplots

    plt.style.use(niceplots.get_style())

    niceColors = niceplots.get_colors()
    TACSMat = readSparseCOOMat("TACSJacobian.mtx")
    KernelMat = readSparseCOOMat("KernelJacobian.mtx")
    markerSize = max(200 / TACSMat.shape[0], 0.1)
    plt.spy(TACSMat, c=niceColors["Yellow"], marker="s", markeredgecolor=niceColors["Yellow"], markersize=markerSize)
    plt.spy(KernelMat, c=niceColors["Blue"], marker="o", markeredgecolor=niceColors["Blue"], markersize=markerSize)

    maxEntry = max(TACSMat.data.max(), KernelMat.data.max())
    minEntry = min(TACSMat.data.min(), KernelMat.data.min())
    maxEntry = max(abs(maxEntry), abs(minEntry))

    fig, axes = plt.subplots(ncols=3, figsize=(12, 6))
    pos = axes[0].matshow(TACSMat.toarray(), cmap="coolwarm", vmin=-maxEntry, vmax=maxEntry)
    fig.colorbar(pos, ax=axes[0])
    axes[0].set_title("TACS Jacobian")

    pos = axes[1].matshow(KernelMat.toarray(), cmap="coolwarm", vmin=-maxEntry, vmax=maxEntry)
    fig.colorbar(pos, ax=axes[1])
    axes[1].set_title("Kernel Jacobian")

    # Compute the difference between the two matrices
    diff = TACSMat - KernelMat
    maxDiff = max(abs(diff.data.max()), abs(diff.data.min()))
    pos = axes[2].matshow(TACSMat.toarray()-KernelMat.toarray(), vmin=-maxDiff, vmax=maxDiff, cmap="coolwarm")
    fig.colorbar(pos, ax=axes[2])
    axes[2].set_title("Difference")

    plt.show()
