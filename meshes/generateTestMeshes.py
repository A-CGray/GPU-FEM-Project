"""
==============================================================================
Test meshes for parallel computing project
==============================================================================
@File    :   generateTestMeshes.py
@Date    :   2024/11/26
@Author  :   Alasdair Christison Gray
@Description : This script generates the meshes I use(d) for my GPU performance testing. For each of the 3 geometries,
I want 1st-4th order meshes with 10^3 to 10^7 DOF.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================

# ==============================================================================
# External Python modules
# ==============================================================================

# ==============================================================================
# Extension modules
# ==============================================================================
from generateAnnularMesh import generateAnnularMesh
from generateLBracketMesh import generateLBracketMesh
from generateSquareMesh import generateSquareMesh

# For each geometry we want a way to compute the mesh size that will give use the desired number of DOF.
# Can try doing this by doing a curve fit of the number of DOF vs mesh size for a first order mesh on each geometry then
# multiplying by a constant to account for higher order meshes.

# If a first order mesh with a given element size has N DOF then:
# - Second order mesh has roughly 4N DOF
# - Third order mesh has roughly 9N DOF
# - Fourth order mesh has roughly 16N DOF
elemOrderFactor = [1, 4, 9, 16]

# Assume that for larger DOF counts, the number of DOF scales with 1/(element size^2)
squareCoeff = 2.317608  # NDOF = elemOrderFactor * squareCoeff * 1/(meshSize^2)
LBracketCoeff = 1.5047119999999998  # NDOF = elemOrderFactor * LBracketCoeff * 1/(meshSize^2)
annulusCoef = 4.707312  # NDOF = elemOrderFactor * annulusCoef * 1/(meshSize^2)

targetDOF = [10**ii for ii in range(3, 8)]

meshData = {}
meshData["Square"] = {"func": generateSquareMesh, "coeff": squareCoeff}
meshData["LBracket"] = {"func": generateLBracketMesh, "coeff": LBracketCoeff}
meshData["Annulus"] = {"func": generateAnnularMesh, "coeff": annulusCoef}

for order in range(1, 5):
    for DOF in targetDOF:
        for name, data in meshData.items():
            meshFunc = data["func"]
            coeff = data["coeff"]
            meshSize = (elemOrderFactor[order - 1] * coeff / DOF) ** 0.5
            print(f"For {name} geometry, {DOF} DOF, order {order} mesh, using mesh size {meshSize}")
            meshFunc(meshSize=meshSize, order=order, visualise=False)
