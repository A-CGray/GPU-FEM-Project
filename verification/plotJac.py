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
    A = readSparseCOOMat("jacobian.mtx")
    markerSize = max(200 / A.shape[0], 0.1)
    plt.spy(A, c=niceColors["Yellow"], marker="s", markeredgecolor=niceColors["Yellow"], markersize=markerSize)
    plt.show()
