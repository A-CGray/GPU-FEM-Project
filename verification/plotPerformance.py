import glob
import os

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import niceplots

def getMeshInfoFromName(fileName:str):
    name = os.path.split(fileName)[-1]
    nameParts = name[:-4].split("-")
    geomType = nameParts[0]
    elementOrder = int(nameParts[1][-1])
    numElements = int(nameParts[2].replace("Elements", ""))
    numDOF = int(nameParts[3].replace("DOF", ""))
    return geomType, elementOrder, numElements, numDOF


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="./", help="Directory to look for results in and to write outputs to")
    args = parser.parse_args()

    plt.style.use(niceplots.get_style())

    plotMarkers = {"Annulus": "o", "Square": "s", "LBracket": "^"}
    plotColors = niceplots.get_colors_list()

    jacTimingFiles = glob.glob(os.path.join(args.dir, "*-JacobianTimings.csv"))
    resTimingFiles = glob.glob(os.path.join(args.dir, "*-ResidualTimings.csv"))

    # Make plots for Jacobian and residual times vs numDOF
    for timingFiles, name in zip([jacTimingFiles, resTimingFiles], ["Jacobian", "Residual"]):
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_xscale("log")
        ax.set_xlabel("Number of DOF")
        ax.set_xscale("log")
        ax.set_ylabel("GDOF/s")
        ax.set_title(f"{name} assembly performance")
        for file in timingFiles:
            geomType, elementOrder, numElements, numDOF = getMeshInfoFromName(file)
            if numDOF > 500:
                times = np.loadtxt(file)
                medianDOFPerSec = numDOF / np.median(times)/1e9

                ax.plot(
                    numDOF,
                    medianDOFPerSec,
                    marker=plotMarkers[geomType],
                    color=plotColors[elementOrder - 1],
                    markersize=10,
                    clip_on=False,
                )

        # For each marker type I want to have a legend entry with an uncolored version of that marker
        geomHandles = []
        geomLabels = []
        for geomType in plotMarkers.keys():
            geomHandles.append(Line2D([0], [0], marker=plotMarkers[geomType], color="k", markersize=10, linestyle=""))
            geomLabels.append(geomType)

        # Then I'll have a second legend entry for the element order
        orderHandles = []
        orderLabels = []
        for i in range(1, 5):
            orderHandles.append(Line2D([0], [0], marker="s", color=plotColors[i-1], linestyle="", markersize=10))
            orderLabels.append(f"{i}")

        niceplots.adjust_spines(ax)
        geomLegend = ax.legend(geomHandles, geomLabels, title="Mesh Geometry:", loc="upper left")
        ax.add_artist(geomLegend)
        ax.legend(orderHandles, orderLabels, title="Element Order:", loc='center left', labelcolor="linecolor")

        niceplots.save_figs(fig, os.path.join(args.dir, f"{name}-Performance"), formats=["pdf", "png"])

