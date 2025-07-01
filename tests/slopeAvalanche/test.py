"""
Test for bedload value
"""

import numpy as np
from fluidfoam import readof as rdf

print(" --- running slope avalanche --- ")

success = True
tolBetaRep = 1e-3  # tolerance with repose angle
tolQav = 1e-4  # tolerance on bedload value, comparison with Vinent (2019)
tolBeta = 1e-5  # error on beta from solver and computed from qav component

time = "0.1"
endTime = "latestTime"

dS = 0.5e-3  # sediment diameter
rhoS = 2650.  # sediment density
rhoF = 1000.  # fluid density
g = 9.81  # gravity acceleration
repAngle = 32.  # repose angle of granular material (in rad)

# bed position
Xbed, Ybed, Zbed = rdf.readmesh(
    "./", endTime, boundary="bed", verbose=False)

tanBed = np.abs((Zbed[2:] - Zbed[:-2]) / (Xbed[2:] - Xbed[:-2]))

maxAngle = (180/np.pi) * np.max(np.arctan(tanBed))

repAngleError = (maxAngle - repAngle) / repAngle
if repAngleError > tolBetaRep:
    success = False
    print(
        f"error! relative difference between repose angle "
        + f"({repAngle} deg) and maximum slope angle ({maxAngle} deg) "
        + f"is {repAngleError*100} %\ntolerance is {tolBetaRep*100} %")
else:
    print(
        f"bed slope OK, maximum slope is {round(maxAngle, 5)}°, "
        + f"repose angle is {repAngle} °")

# test value of avalanche
Qav0 = 5e-3
betaRep = repAngle * np.pi / 180
qavX, qavY, qavZ = rdf.readvector(
    "./", time, "qavVf", boundary="bed", verbose=False)
magQav = np.sqrt(qavX**2 + qavZ**2)
# slope angle from sedExnerFOAM
betaOF = rdf.readscalar("./", time, "betaVf", boundary="bed", verbose=False)
# compute slope angle from qav components
betaPY = np.arctan(np.abs(qavZ/qavX))
qavVinent = Qav0 * (np.tanh(np.tan(betaPY)) - np.tanh(np.tan(betaRep)))
qavVinent /= (1 - np.tanh(np.tan(betaRep)))

# error on avalanche bedload flux mag
relErrQav = np.max(np.abs(magQav - qavVinent) / np.max(qavVinent))
if relErrQav > tolQav:
    success = False
    print(
        f"ERROR! relative error on avalanche bedload flux value: "
        + f"{relErrQav*100} %\ntolerance is {tolQav*100} %")
else:
    print(f"avalanche flux OK, max error {relErrQav}")

# error on slope angle from solver and qav components
relErrBeta = (betaPY - betaOF) / betaOF
if np.any(relErrBeta > tolBeta):
    success = False
    print(
        "ERROR! difference between beta from solver and from qav component"
        + f"\n  max(betaOF): {np.max(betaOF)*180/np.pi}°"
        + f"\n  max(betaPY): {np.max(betaPY)*180/np.pi}°"
        + f"\n  maximum relative error: {np.max(relErrBeta)*100}"
        + f"%\ttolerance: {tolQav*100} %")
else:
    print(f"beta value OK, maximum relative error {np.max(relErrBeta)}")


assert success
