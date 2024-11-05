"""
==============================================================================
L-Bracket mesh generation
==============================================================================
@File    :   generateLBracket.py
@Date    :   2024/11/05
@Author  :   Alasdair Christison Gray
@Description : Uses gmsh to generate a mesh of the classic L-Bracket geometry
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


def generateLBracket(
    sideLength: float = 1.0,
    cutoutSize: float = 0.6,
    meshSize: float = 0.1,
    order: int = 1,
    refine: int = 0,
    visualise: bool = False,
):
    """
    Generate the L-Bracket geometry using gmsh
    """

    # Initialize the gmsh
    gmsh.initialize()

    legWidth = sideLength - cutoutSize
    perimeterPointCoords: list[list[float]] = [
        [0, 0],
        [sideLength, 0],
        [sideLength, legWidth],
        [legWidth, legWidth],
        [legWidth, sideLength],
        [0, sideLength],
    ]

    gmsh.model.add("LBracket")

    surface, _, _, _ = createPlaneSurfaceFromPerimeterPoints(perimeterPointCoords)

    gmsh.model.geo.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1, name="LBracket")

    meshSurface(surface, meshSize=meshSize, order=order, refine=refine)

    gmsh.write("LBracket.msh")
    gmsh.write("LBracket.bdf")

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
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("--visualise", action="store_true")
    args = parser.parse_args()

    generateLBracket(
        sideLength=args.sideLength,
        cutoutSize=args.cutoutSize,
        meshSize=args.meshSize,
        order=args.order,
        refine=args.refine,
        visualise=args.visualise,
    )
