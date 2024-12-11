import glob
import os

import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import niceplots


def getMeshInfoFromName(fileName: str):
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
    parser.add_argument(
        "--dir", type=str, default="./", help="Directory to look for results in and to write outputs to"
    )
    args = parser.parse_args()

    # args.dir = "/run/media/ali/T7 Shield/GPU-FEM-Project/QuadPtParallel"

    plt.style.use(niceplots.get_style())

    plotMarkers = {"Annulus": "o", "Square": "s", "LBracket": "^"}
    plotColors = niceplots.get_colors_list()

    jacTimingFiles = glob.glob(os.path.join(args.dir, "*-JacobianTimings.csv"))
    resTimingFiles = glob.glob(os.path.join(args.dir, "*-ResidualTimings.csv"))

    # Make plots for Jacobian and residual times vs numDOF
    for timingFiles, name in zip([jacTimingFiles, resTimingFiles], ["Jacobian", "Residual"]):
        efFig, efAx = plt.subplots(figsize=(10, 10))
        efAx.set_xscale("log")
        efAx.set_xlabel("Number of DOF")
        efAx.set_ylabel("GDOF/s")
        efAx.set_title(f"{name} assembly performance")

        timeFig, timeAx = plt.subplots(figsize=(10, 10))
        timeAx.set_xscale("log")
        timeAx.set_xlabel("Number of DOF")
        timeAx.set_yscale("log")
        timeAx.set_ylabel("Median wall time (s)")
        timeAx.set_title(f"{name} assembly performance")
        for file in timingFiles:
            geomType, elementOrder, numElements, numDOF = getMeshInfoFromName(file)
            if numDOF > 500:
                times = np.loadtxt(file)
                medianTime = np.median(times)
                medianDOFPerSec = numDOF / medianTime / 1e9

                efAx.plot(
                    numDOF,
                    medianDOFPerSec,
                    marker=plotMarkers[geomType],
                    color=plotColors[elementOrder - 1],
                    markersize=10,
                    clip_on=False,
                )
                timeAx.plot(
                    numDOF,
                    medianTime,
                    marker=plotMarkers[geomType],
                    color=plotColors[elementOrder - 1],
                    markersize=10,
                    clip_on=False,
                )
        efAx.set_ylim(bottom=0)
        efAx.set_xlim(1e3, 1e7)
        timeAx.set_xlim(1e3, 1e7)

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
            orderHandles.append(Line2D([0], [0], marker="s", color=plotColors[i - 1], linestyle="", markersize=10))
            orderLabels.append(f"{i}")

        for fig, ax, measureName in zip([efFig, timeFig], [efAx, timeAx], ["Efficiency", "Time"]):
            niceplots.adjust_spines(ax)
            geomLegend = ax.legend(geomHandles, geomLabels, title="Mesh Geometry:", loc="upper left")
            ax.add_artist(geomLegend)
            ax.legend(orderHandles, orderLabels, title="Element Order:", loc="center left", labelcolor="linecolor")

            niceplots.save_figs(fig, os.path.join(args.dir, f"{name}-{measureName}"), formats=["pdf", "png"])
