"""
==============================================================================
utility functions for making meshes with gmsh
==============================================================================
@File    :   GmshUtils.py
@Date    :   2024/11/05
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
from typing import Sequence, Optional

# ==============================================================================
# External Python modules
# ==============================================================================
import gmsh
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================


def createPlaneSurfaceFromPerimeterPoints(perimeterPointCoords: Sequence[Sequence[float]]):
    # Create gmsh points

    perimeterPoints: list[int] = []
    for ii, coord in enumerate(perimeterPointCoords):
        perimeterPoints.append(gmsh.model.geo.addPoint(coord[0], coord[1], 0))

    # Create gmsh lines
    perimeterLines: list[int] = []
    numPoints = len(perimeterPoints)
    for ii in range(numPoints):
        perimeterLines.append(gmsh.model.geo.addLine(perimeterPoints[ii], perimeterPoints[(ii + 1) % numPoints]))

    # Create the loop of curves defining the perimeter and use it to create a plane surface
    loop = gmsh.model.geo.addCurveLoop(perimeterLines)
    surface = gmsh.model.geo.addPlaneSurface([loop])
    return surface, loop, perimeterLines, perimeterPoints


def meshSurface(
    surface: int,
    meshSize: float = 0.1,
    order: int = 1,
    refine: int = 0,
    smoothingIterations: int = 10,
):
    gmsh.option.setNumber("General.NumThreads", 8)

    # Create a uniform sizing field
    def meshSizeCallback(dim, tag, x, y, z, lc):
        return meshSize

    gmsh.model.mesh.setSizeCallback(meshSizeCallback)

    # Set meshing options and create mesh
    # To generate quadrangles instead of triangles, we can simply add
    gmsh.model.mesh.setRecombine(2, surface)

    # The default recombination algorithm is called "Blossom": it uses a minimum
    # cost perfect matching algorithm to generate fully quadrilateral meshes from
    # triangulations. More details about the algorithm can be found in the
    # following paper: J.-F. Remacle, J. Lambrechts, B. Seny, E. Marchandise,
    # A. Johnen and C. Geuzaine, "Blossom-Quad: a non-uniform quadrilateral mesh
    # generator using a minimum cost perfect matching algorithm", International
    # Journal for Numerical Methods in Engineering 89, pp. 1102-1119, 2012.

    # For even better 2D (planar) quadrilateral meshes, you can try the
    # experimental "Frontal-Delaunay for quads" meshing algorithm, which is a
    # triangulation algorithm that enables to create right triangles almost
    # everywhere: J.-F. Remacle, F. Henrotte, T. Carrier-Baudouin, E. Bechet,
    # E. Marchandise, C. Geuzaine and T. Mouton. A frontal Delaunay quad mesh
    # generator using the L^inf norm. International Journal for Numerical Methods
    # in Engineering, 94, pp. 494-512, 2013. Uncomment the following line to try
    # the Frontal-Delaunay algorithms for quads:
    #
    # 1: MeshAdapt, 2: Automatic, 3: Initial mesh only, 5: Delaunay, 6: Frontal-Delaunay, 7: BAMG, 8: Frontal-Delaunay for Quads, 9: Packing of Parallelograms, 11: Quasi-structured Quad
    gmsh.option.setNumber("Mesh.Algorithm", 1)

    # The default recombination algorithm might leave some triangles in the mesh, if
    # recombining all the triangles leads to badly shaped quads. In such cases, to
    # generate full-quad meshes, you can either subdivide the resulting hybrid mesh
    # (with `Mesh.SubdivisionAlgorithm' set to 1), or use the full-quad
    # recombination algorithm, which will automatically perform a coarser mesh
    # followed by recombination, smoothing and subdivision. Uncomment the following
    # line to try the full-quad algorithm:
    #
    # 0: simple, 1: blossom, 2: simple full-quad, 3: blossom full-quad
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)  # or 3

    # You can also set the subdivision step alone, with
    #
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)

    gmsh.option.setNumber("Mesh.Smoothing", 100)
    gmsh.option.setNumber("Mesh.ElementOrder", order)

    gmsh.model.mesh.generate(2)

    if refine > 0:
        for _ in range(refine):
            gmsh.model.mesh.refine()

        # Refining the mesh puts it back to first order elements, need to re-mesh to get back to desired order
        if order > 1:
            gmsh.model.mesh.setOrder(order)
            gmsh.model.mesh.generate(2)

    smoothingType = "Laplace2D" if order == 1 else "HighOrder"
    gmsh.model.mesh.optimize(smoothingType, niter=smoothingIterations)


def fixGmshBDF(inputFile: str, outputFile: Optional[str] = None):
    # Open the mesh file and a new file with the same name but with "Fixed" appended to the end
    with open(inputFile, "r") as f:
        origLines = f.readlines()
        newLines = computeNewLines(origLines)

    # Now write the new lines to the new file
    if outputFile is None:
        outputFile = inputFile.replace(".bdf", "Fixed.bdf")
    with open(f"{outputFile}", "w") as g:
        for line in newLines:
            if line[-1] != "\n":
                line += "\n"
            g.write(line)


def parseElementLines(elementLines: str):
    columnWidth = 8
    elementLine = []
    lineEndTerm = None
    for line in elementLines:
        # Remove the newline character and potentially the "+" term that gmsh adds
        line = line.replace("\n", "")
        if lineEndTerm is not None:
            line = line.replace(lineEndTerm, "")
        # See if there's a "+" character in the line and if so, remove everything after it
        plusPosition = line.find("+")
        if plusPosition != -1:
            lineEndTerm = line[plusPosition:]
            line = line[:plusPosition]
        # Go through the line and split it into 8 character terms
        elementLine += [line[i : i + columnWidth] for i in range(0, len(line), columnWidth)]
    return elementLine


def computeNewLines(origLines):
    newLines = []
    notFinished = True
    lineNum = 0
    while notFinished:
        line = origLines[lineNum]
        if "CQUAD" in line:
            # Combine all the lines up to the next "ENDDATA" or "CQUAD4"
            elementLines = [line]
            lineNum += 1
            while "CQUAD" not in origLines[lineNum] and "ENDDATA" not in origLines[lineNum]:
                elementLines.append(origLines[lineNum])
                lineNum += 1
            elementLine = parseElementLines(elementLines)

            # # Combine the lines into a single line then split into a list by splitting every 8 characters
            # elementLine = "".join(elementLines).replace("\n", "")
            # elementLine = [elementLine[i : i + columnWidth] for i in range(0, len(elementLine), columnWidth)]
            # # Remove the weird + symbol terms that gmsh adds and any newlines
            # elementLine = [term for term in elementLine if "+" not in term]
            #
            # Figure out which element type we have
            numNodes = len(elementLine) - 3
            # Raise an error if the number of nodes is not a square number
            if np.sqrt(numNodes) % 1 != 0:
                raise ValueError(f"Found element with {numNodes} nodes, which is not a square number")
            elementLine[0] = f"CQUAD{numNodes}"
            # Write the new line(s) to the new file
            entryCount = 0
            newLine = ""
            for term in elementLine:
                newLine += f"{term}".ljust(8)
                entryCount += 1
                if entryCount % 9 == 0:
                    newLine += "\n" + " " * 8
                    entryCount += 1
            newLines.append(newLine)

        else:
            newLines.append(line)
            lineNum += 1
            if "ENDDATA" in line:
                notFinished = False
    return newLines
