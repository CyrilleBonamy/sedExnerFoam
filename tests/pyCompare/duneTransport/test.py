"""
from saturated bedload flux, compute the actual bedload
compare with results from simulation
"""

import numpy as np
from fluidfoam import readof as rdf
import os


print("\n --- running test dune transport --- ")

success = True
tolQb = 1e-5

# physical parameters
dS = 0.4e-3  # sediment diameter
rhoS = 2500.  # sediment density
rhoF = 1000.  # fluid density
nuF = 1e-6  # fluid kinematic viscosity
g = 9.81  # gravity acceleration

Tsat = 0.05  # saturation time
Lsat = 5e-3  # saturation length

foamTimes = os.popen("foamListTimes").read()
timeList = foamTimes.split('\n')[:-1]
ntimes = len(timeList)

maxRelErrQb_alldt = np.zeros(ntimes)

for i in range(ntimes-1):
    t0, t = timeList[i], timeList[i+1]
    dt = round(float(t) - float(t0), 5)
    Xb, Yb, Zb = rdf.readmesh("./", t, boundary="bed", verbose=False)
    dX = Xb[1] - Xb[0]  # mesh is uniform in x-direction

    # saturated bedload
    qsX, qsY, qsZ = rdf.readvector(
        "./", t, "qsatVf", boundary="bed", verbose=False)

    # current time bedload qb, induced by shear stress
    qbX, qbY, qbZ = rdf.readvector(
        "./", t, "qbVf", boundary="bed", verbose=False)

    # previous time bedload qb, induced by shear stress
    qb0X, qb0Y, qb0Z = rdf.readvector(
        "./", t0, "qbVf", boundary="bed", verbose=False)

    qbE = np.zeros(len(Xb) + 1)
    # old qb on edges, linear interpolation on uniform mesh
    qbE[1:-1] = 0.5 * (qb0X[1:] + qb0X[:-1])
    qbE[0], qbE[-1] = qb0X[0], qb0X[-1]  # zero gradient BC
    eFqb = qbE
    # compute current bedload using saturation equation, x-component
    qbPY = qb0X + (dt / Tsat) * (
        qsX - qb0X + (Lsat / dX) * (eFqb[:-1] - eFqb[1:]))

    errQb = (qbX - qbPY) / np.max(np.abs(qbPY))
    maxRelErrQb = np.max(np.abs(errQb))
    maxRelErrQb_alldt[i] = maxRelErrQb

maxError = np.max(maxRelErrQb_alldt)
if maxError > tolQb:
    success = False
    print(
        "ERROR! qb from saturation equation in sedExnerFoam "
        + "not matching qb computed in post process"
        + f"\n relative error is {maxError}"
        + f"\n tolerance is {tolQb}")
else:
    print(f"qb from saturation: OK, max error: {maxError}")


assert success
