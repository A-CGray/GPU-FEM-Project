"""
==============================================================================
Square mesh generation
==============================================================================
@File    :   generateSquareMesh.py
@Date    :   2024/11/05
@Author  :   Alasdair Christison Gray
@Description : Uses gmsh to generate a mesh of a basic square
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================
import gmsh

# ==============================================================================
# Extension modules
# ==============================================================================
from GmshUtils import meshSurface, createPlaneSurfaceFromPerimeterPoints


def generateSquareMesh(
    sideLength: float = 1.0,
    meshSize: float = 0.1,
    order: int = 1,
    refine: int = 0,
    smoothingIterations: int = 10,
    visualise: bool = False,
):
    """
    Generate the L-Bracket geometry using gmsh
    """

    # Initialize the gmsh
    gmsh.initialize()

    perimeterPointCoords: list[list[float]] = [
        [0, 0],
        [sideLength, 0],
        [sideLength, sideLength],
        [0, sideLength],
    ]

    gmsh.model.add("Square")

    surface, _, _, _ = createPlaneSurfaceFromPerimeterPoints(perimeterPointCoords)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1, name="Square")

    meshSurface(surface, meshSize=meshSize, order=order, refine=refine, smoothingIterations=smoothingIterations)

    nodeIDs, nodeCoords, _ = gmsh.model.mesh.getNodes()
    elementTypes, elementTags, elementNodes = gmsh.model.mesh.getElements(2)
    numNodes = len(nodeIDs)
    numElements = len(elementTags[0])

    meshName = f"Square-Order{order}-{numElements}Elements-{numNodes*2}DOF"

    gmsh.write(f"{meshName}.msh")
    gmsh.write(f"{meshName}.bdf")

    if visualise:
        gmsh.option.setNumber("Mesh.Nodes", 1)
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sideLength", type=float, default=1.0)
    parser.add_argument("--cutoutSize", type=float, default=0.6)
    parser.add_argument("--meshSize", type=float, default=0.1)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("--visualise", action="store_true")
    args = parser.parse_args()

    generateSquareMesh(
        sideLength=args.sideLength,
        meshSize=args.meshSize,
        order=args.order,
        refine=args.refine,
        smoothingIterations=args.smooth,
        visualise=args.visualise,
    )
