import sympy as sp
import numpy as np

def getLagrangePolyCoeffs(order):
    x = sp.symbols("x")
    knots = np.linspace(-1,1, order+1)

    print(f"\n\nComputing coefficients for order {order} Lagrange polynomial")
    for nodeInd in range(order+1):
        xNode = knots[nodeInd]
        L = 1.0
        for ii in range(order+1):
            if ii!=nodeInd:
                xi = knots[ii]
                L *= (x-xi)/(xNode-xi)
        coeffs = sp.Poly(L).all_coeffs()
        print(f"Node {nodeInd}: {coeffs=}")

for order in range(1,5):
    getLagrangePolyCoeffs(order)
