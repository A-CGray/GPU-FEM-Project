"""
==============================================================================

==============================================================================
@File    :   generateAnnularMesh.py
@Date    :   2024/11/05
@Author  :   Alasdair Christison Gray
@Description :
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
from GmshUtils import meshSurface


def generateAnnularMesh(
    outerRadius: float = 1.0,
    innerRadius: float = 0.6,
    meshSize: float = 0.1,
    order: int = 1,
    refine: int = 0,
    smoothingIterations: int = 10,
    visualise: bool = False,
):
    gmsh.initialize()

    outerCircle = gmsh.model.occ.addCircle(0, 0, 0, outerRadius)
    innerCircle = gmsh.model.occ.addCircle(0, 0, 0, innerRadius)

    outerLoop = gmsh.model.occ.addCurveLoop([outerCircle])
    innerLoop = gmsh.model.occ.addCurveLoop([innerCircle])
    surface = gmsh.model.occ.addPlaneSurface([outerLoop, innerLoop])

    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [surface], 1, name="Annulus")

    meshSurface(surface, meshSize=meshSize, order=order, refine=refine, smoothingIterations=smoothingIterations)

    nodeIDs, nodeCoords, _ = gmsh.model.mesh.getNodes(includeBoundary=True)
    elementTypes, elementTags, elementNodes = gmsh.model.mesh.getElements(2)
    numNodes = len(nodeIDs)
    numElements = len(elementTags[0])

    meshName = f"Annulus-Order{order}-{numElements}Elements-{numNodes*2}DOF"

    gmsh.write(f"{meshName}.bdf")

    if visualise:
        gmsh.option.setNumber("Mesh.Nodes", 1)
        gmsh.option.setNumber("Mesh.SurfaceFaces", 1)
        gmsh.fltk.run()

    gmsh.finalize()

    return meshName


if __name__ == "__main__":
    import argparse
    from GmshUtils import fixGmshBDF

    parser = argparse.ArgumentParser()
    parser.add_argument("--outerRad", type=float, default=1.0)
    parser.add_argument("--innerRad", type=float, default=0.6)
    parser.add_argument("--meshSize", type=float, default=0.1)
    parser.add_argument("--order", type=int, default=1)
    parser.add_argument("--smooth", type=int, default=10)
    parser.add_argument("--refine", type=int, default=0)
    parser.add_argument("--visualise", action="store_true")
    args = parser.parse_args()

    meshName = generateAnnularMesh(
        outerRadius=args.outerRad,
        innerRadius=args.innerRad,
        meshSize=args.meshSize,
        order=args.order,
        refine=args.refine,
        smoothingIterations=args.smooth,
        visualise=args.visualise,
    )

    if args.order > 1:
        fixGmshBDF(f"{meshName}.bdf")
