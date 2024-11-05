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
from typing import Sequence

# ==============================================================================
# External Python modules
# ==============================================================================
import gmsh

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
    # gmsh.option.setNumber("Mesh.Algorithm", 8)

    # The default recombination algorithm might leave some triangles in the mesh, if
    # recombining all the triangles leads to badly shaped quads. In such cases, to
    # generate full-quad meshes, you can either subdivide the resulting hybrid mesh
    # (with `Mesh.SubdivisionAlgorithm' set to 1), or use the full-quad
    # recombination algorithm, which will automatically perform a coarser mesh
    # followed by recombination, smoothing and subdivision. Uncomment the following
    # line to try the full-quad algorithm:
    #
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 3)  # or 3

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
        gmsh.model.mesh.generate(2)

    smoothingType = "Laplace2D" if order == 1 else "HighOrder"
    gmsh.model.mesh.optimize(smoothingType, niter=smoothingIterations)
