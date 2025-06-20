"""
Check mass conservation, between suspended load and bed during deposition
need all time steps saved
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import fluidfoam.readof as rdf

plt.rcParams["font.size"] = 12


# load previous results to track change in solver behavior
zpr, *csAllTime = np.loadtxt(
    "./dataSed.txt", unpack=True, delimiter=";")


def myReadmesh(path, t):
    Zcells = rdf.readmesh(path, time_name=t, verbose=False)[2]
    ncells = len(Zcells)
    zbed = rdf.readmesh(
        path, time_name=t, boundary="bed", verbose=False)[2][0]
    ztop = rdf.readmesh(
        path, time_name=t, boundary="top", verbose=False)[2][0]
    Zfaces = np.zeros(ncells+1)
    print(zbed, ztop)
    Zfaces[0], Zfaces[-1] = zbed, ztop
    for i in range(1, ncells):
        Zfaces[i] = 1.5 * Zcells[i-1] - 0.5 * Zfaces[i-1]
    return Zcells, Zfaces


foamTimes = os.popen("foamListTimes -withZero").read()

timeList = foamTimes.split('\n')[:-1]
timeArr = np.array([float(t) for t in timeList])
ntimes = len(timeList)

# color list for Cs results
colors = ["cornflowerblue", "tomato"]
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=ntimes)
colors = cm(np.arange(0, cm.N))

rhoS = 2600.  # sediment density
CsMax = 0.6

# domain and mesh dimensions
domHeight = 1.
domWidth = 0.1
# print(rdf.readmesh("./", time_name="10"))
ncells = len(rdf.readmesh("./", verbose=False)[2])

bedArea = domWidth**2

print("-domain dimensions")
print(f"height {domHeight} m")
print(f"area {round(bedArea, 5)} m2")
print(f"number of cells {ncells}")

massSuspension = np.zeros(ntimes-1)
massBed = np.zeros(ntimes-1)
totMass = np.zeros(ntimes-1)
relMassErr = np.zeros(ntimes-1)

# read initial suspended sediment volume fraction
Cs0 = rdf.readscalar("./", "0", "Cs", verbose=False)[0]
# read settling velocity
ws = - rdf.readvector("./", "latestTime", "Ws", verbose=False)[2, 0]
# initial bed velocity
wBed = 0.

# compute theoretical end time of settling
wBedTheo = ws * CsMax / (CsMax - Cs0)
TendSettle = domHeight / (ws + wBedTheo)
# error on Cs, compare with previous simulation
errCs = np.zeros(ntimes-1)

fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(2, 2)

axM = fig.add_subplot(gs[0, 0])
axErrM = fig.add_subplot(gs[1, 0])
axCs = fig.add_subplot(gs[0, 1])
axErrCs = fig.add_subplot(gs[1, 1])

Zcells, Zfaces = myReadmesh("./", "0")

for i in range(ntimes-1):
    t, tnext = timeList[i], timeList[i+1]
    dt = float(tnext) - float(t)
    print(f"\ntime {t}")
    if i == 0:
        zbed_t = 0.  # initial bed position
    else:
        zbed_t = Zfaces[0]
    print(f"current bed position, zbed(t) = {zbed_t} m")

    # get suspended mass
    Cs = rdf.readscalar("./", time_name=t, name="Cs", verbose=False)
    if Cs.shape == (1,):
        Cs = np.ones(ncells) * Cs
    axCs.plot(Cs, Zcells, color=colors[i])
    try:
        cellVolumes = rdf.readscalar("./", t, "V")
    except FileNotFoundError:
        os.system("postProcess -func writeCellVolumes")
        cellVolumes = rdf.readscalar("./", t, "V")
    if cellVolumes.shape == (1,):
        cellVolumes = np.ones(ncells) * cellVolumes
    massSuspension[i] = rhoS * np.sum(Cs * cellVolumes)
    # get bed mass
    Zcells, Zfaces = myReadmesh("./", tnext)
    if i > 0:
        # mass leaving domain in dt, flom ws Cs flux through bed
        massOut = Cs[0] * rhoS * (ws+wBed) * bedArea * dt
        # mass difference with previous time step
        dSuspMass = massSuspension[i-1] - massSuspension[i]
        print(f"mass leaving domain, {massOut} kg")
        print(f"difference in suspended mass, {dSuspMass} kg")
        print(f"difference, {massOut-dSuspMass} kg")

    # update bed velocity
    wBed = (Zfaces[0] - zbed_t) / dt
    print(f"bed position t+dt, zbed(t+dt) = {Zfaces[0]} m")
    print(f"bed velocity, wbed = {wBed} m/s")

    # bed is only one face, mesh is 1D
    massBed[i] = rhoS * CsMax * Zfaces[0] * bedArea
    if i > 0:
        dMbed = massBed[i]-massBed[i-1]
        print(f"mass entering bed, {dMbed}")
    print(f"mass in suspension: {massSuspension[i]} kg")
    print(f"bed mass: {massBed[i]} kg")
    # compare Cs field with previous results
    csPrev = csAllTime[i]
    errCs[i] = np.max(np.abs(csPrev - Cs) / Cs0)


totMass = massBed + massSuspension
# relative mass gain or loss
relMassErr = np.abs(totMass - totMass[0]) / totMass[0]

# total mass of sediments
print(f"initial mass: {round(totMass[0], 5)} kg, "
      + f"final mass: {round(totMass[-1], 5)} kg")

axM.plot(
    timeArr[:-1], totMass, ls="solid", lw=1.8, color="#0072B2", label="total")
axM.plot(
    timeArr[:-1], massSuspension, ls="dashed", lw=2,
    color="#009E71", label="suspension")
axM.plot(
    timeArr[:-1], massBed, ls="dashdot", lw=2, color="#D55E00", label="bed")
axM.axvline(TendSettle, color="black", ls="dashed")

axM.set_ylabel("mass [kg]")
axM.grid()
axM.legend(fontsize=15)
axM.tick_params(
    axis="both", which="major")

axErrM.plot(timeArr[:-1], relMassErr, color="#0072B2")
axErrM.set_ylabel("relative error on mass")
axErrM.set_xlabel("time [s]")
axErrM.tick_params(
    axis="both", which="major")
axErrM.grid()

axCs.set_ylabel("z [m]")
axCs.set_xlabel(r"$c_s$")

axErrCs.plot(timeArr[:-1], errCs, marker="x", color="#0072B2")
axErrCs.set_xlabel("time s")
axErrCs.set_ylabel(r"relative error on $c_s$")

fig.tight_layout()

plt.show()
