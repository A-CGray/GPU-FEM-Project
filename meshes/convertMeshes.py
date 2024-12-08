import glob
from GmshUtils import fixGmshBDF

meshName = "Square-Order4-301986Elements-9671234DOF.bdf"
meshNames = glob.glob("*DOF.bdf")

for meshName in meshNames:
    print(f"Fixing {meshName}")
    fixGmshBDF(meshName)
