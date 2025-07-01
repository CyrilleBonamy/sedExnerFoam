"""
compare bed level position with results from previous simulation
compare bed slope angle obtained with different methods:
    - obtained from solver
    - computed from components of qsat, qb, qav, shields
check value of slope correction on Shields number
"""

import numpy as np
from fluidfoam import readof as rdf


print("\n --- running test dune transport --- ")

time = "0.1"

success = True

tolZb = 1e-6
tolSh = 1e-4
tolQs = 1e-4
tolQav = 1e-4
tolBeta = 1e-5

# physical parameters

dS = 0.4e-3  # sediment diameter
rhoS = 2500.  # sediment density
rhoF = 1000.  # fluid density
nuF = 1e-6  # fluid kinematic viscosity
g = 9.81  # gravity acceleration
betaRepDeg = 28.  # granular material repose angle in degrees
betaRep = betaRepDeg * np.pi / 180

einNum = np.sqrt((rhoS/rhoF - 1) * g * dS**3)

Dstar = dS * ((rhoS/rhoF - 1)*g / nuF**2)**(1/3)

critShSoulsby = (0.3 / (1 + 1.2*Dstar)) + 0.055*(1 - np.exp(-0.02*Dstar))

Xb, Yb, Zb = rdf.readmesh("./", time, boundary="bed", verbose=False)
# load xb/zb values from reference simulation
XbPrev, ZbPrev = np.loadtxt(
    "./dataZb01.txt", unpack="True", delimiter=";")
maxRelErrZb = np.max(np.abs(Zb - ZbPrev) / np.max(ZbPrev))
if maxRelErrZb > tolZb:
    success = False
    print("error! bed position differs from previous results"
          + f"\n relative error is {maxRelErrZb}"
          + f"\n tolerance is {tolZb} %")
else:
    print(f"bed position OK, max relative error: {maxRelErrZb}")

# bed slope angle from solver
betaOF = rdf.readscalar("./", time, "betaVf", boundary="bed", verbose=False)

critShields = rdf.readscalar(
    "./", time, "critShieldsVf", boundary="bed", verbose=False)
if critShields.shape == (1,):
    critShields = np.ones_like(Xb) * critShields
shX, shY, shZ = rdf.readvector(
    "./", time, "shieldsVf", boundary="bed", verbose=False)
magSh = np.sqrt(shX**2 + shZ**2)
# compute slope angle beta from Shields components
betaSh = np.arctan(shZ / shX)

# compute slope correction for critical Shields number
cosAlpha = np.where(shX*betaSh > 0, -1, 1)
# limit beta values  to betaRep
betaEff = np.where(np.abs(betaOF) < betaRep, np.abs(betaOF), betaRep)
slopeCorr = np.cos(betaEff) - cosAlpha * np.sin(betaEff)/np.tan(betaRep)
critShCorr = critShSoulsby * slopeCorr

# relative error on critical Shields number
# with slope Correction
maxErrCritShields = np.max((critShields - critShCorr)/np.max(critShCorr))
if maxErrCritShields > tolSh:
    success = False
    print("error on critical Shields values"
          + f"\n relative error is {maxErrCritShields}"
          + f"\n tolerance is {tolSh}")
else:
    print(f"critical Shields value OK, max error {maxErrCritShields}")


# read saturated bedload
qsX, qsY, qsZ = rdf.readvector(
    "./", time, "qsatVf", boundary="bed", verbose=False)
magQs = np.sqrt(qsX**2 + qsZ**2)
betaQsat = np.arctan(qsZ / qsX)

# bedload from Meyer-Peter & Müller formula
qbMPM = einNum * 8 * np.where(
    magSh > critShields, magSh - critShields, 0)**1.5
maxRelErrQs = np.max(np.abs((magQs - qbMPM) / (qbMPM + 1e-8)))

if maxRelErrQs > tolQs:
    success = False
    print("ERROR! saturated bedload values not matching"
          + f"\n relative error is {maxRelErrQs}"
          + f"\n tolerance is {tolQs}")
else:
    print(f"saturated bedload values OK, max error: {maxRelErrQs}")


# read avalanche related bedload flux
qavX, qavY, qavZ = rdf.readvector(
    "./", time, "qavVf", boundary="bed", verbose=False)
magQav = np.sqrt(qavX**2 + qavZ**2)
# mask, positions without avalanche are masked
qavXmask = np.ma.masked_where(np.abs(qavX) < 1e-10, qavX)
qavZmask = np.ma.masked_where(np.abs(qavX) < 1e-10, qavZ)
betaQav = np.arctan(qavZmask / qavXmask)

# bedload due to avalanche from Vinent et. al (2019) formula
Qav0 = 5e-3
qavPY = Qav0 * np.where(
    np.abs(betaOF) > betaRep,
    (np.tanh(np.tan(np.abs(betaOF)))-np.tanh(np.tan(betaRep)))
    / (1-np.tanh(np.tan(betaRep))), 0)
maxRelErrQav = np.max(np.abs((magQav - qavPY) / np.max(qavPY)))

if maxRelErrQav > tolQav:
    success = False
    print("ERROR! avalanche bedload values not matching"
          + f"\n relative error is {maxRelErrQav}"
          + f"\n tolerance is {tolQav}")
else:
    print(f"avalanche bedload values OK, max error: {maxRelErrQav}")

maxErrBetaSh = np.max(
    np.abs(
        (np.abs(betaSh) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))
)
if maxErrBetaSh > tolBeta:
    success = False
    print(
        "ERROR! bed slope computed from Shields "
        + "not matching beta bed slope from solver"
        + f"\n relative error is {maxErrBetaSh}"
        + f"\n tolerance is {tolBeta}")
else:
    print("bed slope angle from Shields components OK, "
          + f"max error: {maxErrBetaSh}")

maxErrBetaQs = np.max(
    np.abs(
        (np.abs(betaQsat) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))
)
if maxErrBetaQs > tolBeta:
    success = False
    print(
        "ERROR! bed slope computed from saturated bedload "
        + "not matching beta bed slope from solver"
        + f"\n relative error is {maxErrBetaQs}"
        + f"\n tolerance is {tolBeta}")
else:
    print("bed slope angle from saturated bedload components OK, "
          + f"max error: {maxErrBetaQs}")

maxErrBetaQav = np.max(
    np.abs(
        (np.abs(betaQav) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))
)
if maxErrBetaQav > tolBeta:
    success = False
    print(
        "ERROR! bed slope computed from savalanche bedload "
        + "not matching beta bed slope from solver"
        + f"\n relative error is {maxErrBetaQav}"
        + f"\n tolerance is {tolBeta}")
else:
    print("bed slope angle from saturated bedload components OK, "
          + f"max error: {maxErrBetaQav}")

assert success
