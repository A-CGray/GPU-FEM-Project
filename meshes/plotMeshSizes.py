import glob
import os

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import niceplots


from GmshUtils import fixGmshBDF

plt.style.use(niceplots.get_style())

plotMarkers = {"Annulus": "o", "Square": "s", "LBracket": "^"}
plotColors = niceplots.get_colors_list()

meshFiles = glob.glob("/nobackup/achris10/GPU-FEM-Project/meshes/*.bdf")

fig, ax = plt.subplots(figsize=(10,10))
ax.set_xscale("log")
ax.set_xlabel("Number of Elements")
ax.set_yscale("log")
ax.set_ylabel("Number of DOF")
for meshFile in meshFiles:
    name = os.path.split(meshFile)[-1]
    nameParts = name[:-4].split("-")
    geomType = nameParts[0]
    elementOrder = int(nameParts[1][-1])
    numElements = int(nameParts[2].replace("Elements", ""))
    numDOF = int(nameParts[3].replace("DOF", ""))
    ax.plot(
        numElements,
        numDOF,
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
ax.legend(orderHandles, orderLabels, title="Element Order:", loc="lower right", labelcolor="linecolor")

niceplots.save_figs(fig, "MeshSizes", formats=["pdf", "png"])
