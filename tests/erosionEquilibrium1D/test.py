"""
1D simulation of erosion and deposition
test if sediment mass is conserved
"""

import numpy as np
from fluidfoam import readof as rdf
import os

print(" --- running 1D erosion test --- ")

success = True
tolMass = 1e-6
tolCritSh = 1e-3
tolQb = 1e-5

foamTimes = os.popen("foamListTimes -withZero").read()
timeList = foamTimes.split("\n")[:-1]
timeArr = np.array([float(t) for t in timeList])
ntimes = len(timeList)

# simulation parameters
nuF = 1e-6  # fluid kinematic viscosity
dS = 0.19e-3  # sediment diameter
rhoS = 2650.  # sediment density
rhoF = 1000.  # water density
CsMax = 0.57  # maximum sediment volumic fraction
g = 9.81  # gravity acceleration
betaRep = 32 * np.pi / 180
# dimless diameter
Dstar = dS * (((rhoS/rhoF - 1) * g) / nuF**2)**(1/3)
# Einstein number
EinNum = np.sqrt((rhoS/rhoF - 1) * g * dS**3)


def qbVanRijn(sh, csh):
    """van Rijn bedload formula"""
    Tshear = (sh / csh - 1)
    return EinNum * 0.053 * Tshear**2.1 / Dstar**0.3


# domain and mesh dimensions
domHeight = 1.
dx, dy = 0.01, 0.001

bedArea = dx * dy

Msusp = np.zeros(ntimes)  # suspended mass
Mbed = np.zeros(ntimes)  # bed mass

Zmesh = rdf.readmesh("./", "latestTime", verbose=False)[2]
nCells = len(Zmesh)

critShields = rdf.readscalar(
    "./", "latestTime", "critShieldsVf",
    boundary="bed", verbose=False)[0]


critShMiedema = (0.2285 / Dstar**1.02) + 0.0575 * (
    1 - np.exp(-0.0225*Dstar))

# Shields number
shields = rdf.readvector(
    "./", "10", "shieldsVf", boundary="bed", verbose=False)[0, 0]

# bedload
qb = rdf.readvector(
    "./", "10", "qbVf", boundary="bed", verbose=False)[0, 0]

qbVR = qbVanRijn(shields, critShields)

# relative error on critical Shields
errCritSh = (critShields - critShMiedema) / critShMiedema
if errCritSh > tolCritSh:
    success = False
    print(
        "error! maximum relative error on critical Shields: "
        + f"{100*errCritSh} %")
    print(f"tolerance is {100*tolCritSh} %")
else:
    print(f"critical Shields value OK, relative error {errCritSh}")

# relative error on bedload
errQb = np.abs((qb - qbVR) / qbVR)
if errQb > tolQb:
    success = False
    print(
        "error! maximum relative error on bedload: "
        + f"{100*errQb} %")
    print(f"tolerance is {100*tolQb} %")
else:
    print(f"bedload value OK, relative error {errQb}")

for i, t in enumerate(timeList):
    zb = rdf.readmesh("./", t, boundary="bed", verbose=False)[2][0]
    Mbed[i] = CsMax * rhoS * zb * dx * dy
    CsField = rdf.readscalar("./", t, "Cs", verbose=False)
    if CsField.shape == (1,):
        CsField = np.ones(nCells) * CsField
    try:
        Vcells = rdf.readscalar(
            "./", t, "V", verbose=False)
    except FileNotFoundError:
        os.system("postProcess -func writeCellVolumes > /dev/null")
        Vcells = rdf.readscalar(
            "./", t, "V", verbose=False)
    Msusp[i] = rhoS * np.sum(CsField * Vcells)

Mtotal = Msusp + Mbed
errMass = np.abs(Mtotal) / np.max(Msusp)
endSimuErr = errMass[-1]  # relative error on mass at t=Tend

# test conservation of mass
if endSimuErr > tolMass:
    success = False
    print(f"error! maximum relative error on mass: {100*endSimuErr} %")
    print(f"tolerance is {100*tolMass} %")
else:
    print(f"mass conservation OK, relative error {endSimuErr}")

assert success
