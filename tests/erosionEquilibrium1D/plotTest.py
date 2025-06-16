"""
Plot 1D erosion test
"""

import numpy as np
from fluidfoam import readof as rdf
import matplotlib.pyplot as plt
import os


plt.rcParams["font.size"] = 15


foamTimes = os.popen("foamListTimes").read()
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
    Tshear = np.where(Tshear > 0, Tshear, 0)
    return EinNum * 0.053 * Tshear**2.1 / Dstar**0.3


# domain and mesh dimensions
domHeight = 1.
dx, dy = 0.01, 0.001

bedArea = dx * dy

Msusp = np.zeros(ntimes)  # suspended mass
Mbed = np.zeros(ntimes)  # bed mass
shields = np.zeros(ntimes)  # Shields number
qbArr = np.zeros(ntimes)  # bedload

Zmesh = rdf.readmesh("./", "latestTime", verbose=False)[2]
nCells = len(Zmesh)

critShields = rdf.readscalar(
    "./", "latestTime", "critShieldsVf",
    boundary="bed", verbose=False)[0]

critShMiedema = (0.2285 / Dstar**1.02) + 0.0575 * (
    1 - np.exp(-0.0225*Dstar))


for i, t in enumerate(timeList):
    zb = rdf.readmesh("./", t, boundary="bed", verbose=False)[2][0]
    Mbed[i] = CsMax * rhoS * zb * dx * dy
    CsField = rdf.readscalar("./", t, "Cs", verbose=False)
    shields[i] = rdf.readvector(
        "./", t, "shieldsVf", boundary="bed", verbose=False)[0][0]
    qbArr[i] = rdf.readvector(
        "./", t, "qbVf", boundary="bed", verbose=False)[0][0]
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


qbVR = qbVanRijn(shields, critShields)
# relative error on critical Shields
errCritSh = (critShields - critShMiedema) / critShMiedema
# relative error on bedload
errQb = np.abs((qbArr - qbVR) / np.max(qbVR))

Mtotal = Msusp + Mbed
errMass = np.abs(Mtotal) / np.max(Msusp)


fig1 = plt.figure(figsize=(10, 8))
gs = fig1.add_gridspec(2, 2)

axSh = fig1.add_subplot(gs[0, 0])
axQb = fig1.add_subplot(gs[0, 1])
axErr = fig1.add_subplot(gs[1, :])

axSh.plot(timeArr, shields)

axSh.plot(
    timeArr, shields, ls="dashed", color="#0072B2", label="OpenFOAM")
axSh.axhline(
    critShields, ls="solid", color="#D55E00", label=r"$\theta_c^0$")
axSh.axhline(
    critShMiedema, ls="dashdot", color="black", label="Miedema")
axSh.set_ylabel(r"$\theta")
axSh.set_xlabel("time s")
axSh.grid()
axSh.legend()

axQb.plot(
    timeArr, qbArr, ls="solid", color="#0072B2", label="OpenFOAM")
axQb.plot(
    timeArr, qbVR, ls="dashed", color="#D55E00", label="van Rijn")
axQb.set_xlabel("time s")
axQb.set_ylabel(r"$q_b\,[m^2.s^{-1}]$")
axQb.grid()
axQb.legend()

axErr.plot(timeArr, errQb, marker="x", color="#0072B2")
axErr.set_xlabel("time s")
axErr.set_ylabel(r"relative error on $q_b$")
axErr.grid()

fig1.tight_layout()


figM, (axMass, axErr) = plt.subplots(nrows=2, figsize=(8.3, 7))

axMass.plot(
    timeArr, Mtotal, ls="solid", color="#0072B2", label="total")
axMass.plot(
    timeArr, Mbed, ls="dashed", color="#D55E00", label="bed")
axMass.plot(
    timeArr, Msusp, ls="dashdot", color="#009E73", label="suspension")

axErr.plot(
    timeArr, errMass, marker="x", color="#0072B2")

axMass.set_ylabel(r"mass $[kg]$")
axMass.grid()
axMass.set_xticklabels([])
axMass.legend()

axErr.set_ylabel("relative mass error")
axErr.grid()
axErr.set_xlabel("time s")

figM.tight_layout()

plt.show()
