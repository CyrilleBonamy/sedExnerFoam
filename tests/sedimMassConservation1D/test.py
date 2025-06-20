"""
sedimentation of sediment in 1D
test if sediment mass is conserved
all time steps must be saved
"""

import numpy as np
from fluidfoam import readof as rdf
import os

print(" --- running sedimentation mass conservation --- ")

success = True
tolCs = 1e-5
tolMass = 1e-5
tolZ = 1e-5

# load previous results to track change in solver behavior
zpr, *csAllTime = np.loadtxt(
    "./dataSed.txt", unpack=True, delimiter=";")

rhoS = 2600.  # sediment density
CsMax = 0.6  # maximum sediment volume fraction

# domain and mesh dimensions
domHeight = 1.
domWidth = 0.1

bedArea = domWidth**2

path = "./"
foamTimes = os.popen("foamListTimes -withZero").read()
timeList = foamTimes.split('\n')[:-1]
ntimes = len(timeList)

Zcells = rdf.readmesh(path, "0", verbose=False)[2]
ncells = len(Zcells)

# initial sediment volume fraction
Cs0 = rdf.readscalar(path, "0", "Cs", verbose=False)[0]
if Cs0 != 0.2:
    success = False
    print(f"initial sediment volume fraction changed, "
          + f"previous value: 0.2  new value: {Cs0}")

if len(zpr) != ncells:
    success = False
    print("error, mesh cell number differs from previous results")

errZ = np.max(zpr - Zcells)/domHeight
if errZ > tolZ:
    success = False
    print("cell position differs from previous results, "
          + f"relative error is {100*zpr} %, "
          + f"tolerance is {100*tol} %")


def myReadMesh(path, t):
    Zcells = rdf.readmesh(path, time_name=t, verbose=False)[2]
    ncells = len(Zcells)
    zbed = rdf.readmesh(
        path, time_name=t, boundary="bed", verbose=False)[2][0]
    ztop = rdf.readmesh(
        path, time_name=t, boundary="top", verbose=False)[2][0]
    Zfaces = np.zeros(ncells+1)
    Zfaces[0], Zfaces[-1] = zbed, ztop
    for i in range(1, ncells):
        Zfaces[i] = 1.5 * Zcells[i-1] - 0.5 * Zfaces[i-1]
    return Zcells, Zfaces


massSuspension = np.zeros(ntimes-1)
massBed = np.zeros(ntimes-1)

for i in range(ntimes-1):
    t, tnext = timeList[i], timeList[i+1]
    dt = float(tnext) - float(t)
    if i == 0:
        zbed_t = 0.  # initial bed position
    else:
        zbed_t = Zfaces[0]

    # get suspended mass
    Cs = rdf.readscalar("./", time_name=t, name="Cs", verbose=False)
    if Cs.shape == (1,):
        Cs = np.ones(ncells) * Cs
    try:
        cellVolumes = rdf.readscalar("./", t, "V", verbose=False)
    except FileNotFoundError:
        os.system("postProcess -func writeCellVolumes > /dev/null")
        cellVolumes = rdf.readscalar("./", t, "V", verbose=False)
    if cellVolumes.shape == (1,):
        cellVolumes = np.ones(ncells) * cellVolumes
    massSuspension[i] = rhoS * np.sum(Cs * cellVolumes)
    # get bed mass
    Zcells, Zfaces = myReadMesh("./", tnext)

    # bed is only one face, mesh is 1D
    massBed[i] = rhoS * CsMax * Zfaces[0] * bedArea
    # compare Cs field with previous results
    csPrev = csAllTime[i]
    errCs = np.max(np.abs(csPrev - Cs) / Cs0)
    if errCs > tolCs:
        success = False
        print(
            f"error! maximum relative error on cs: {100*errCs} %"
            + f"tolerance is {100*tolCs} %")

if success:
    print("Cs values OK, relative error not exceeding {tolCs}")


totMass = massBed + massSuspension
# relative mass gain or loss
maxRelMassErr = np.max(np.abs(totMass - totMass[0]) / totMass[0])

# test conservation of mass
if maxRelMassErr > tolMass:
    success = False
    print(f"error! maximum relative error on mass: {100*relMassErr} %")
    print(f"tolerance is {100*tol} %")
else:
    print(f"mass conservation OK, maximum relative error {maxRelMassErr}")


assert success
