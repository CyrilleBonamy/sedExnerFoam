"""
Plot bed level, bedload flux from simulation
compute saturation and compare with python results
"""

import numpy as np
from fluidfoam import readof as rdf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import os

plt.rcParams["font.size"] = 15
plt.rcParams["lines.linewidth"] = 2.

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
print("time list: ", timeList)
ntimes = len(timeList)

# color list for results
colors = ["cornflowerblue", "tomato"]
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=ntimes-1)
colors = cm(np.arange(0, cm.N))


fig1 = plt.figure(figsize=(12, 7))
gs1 = fig1.add_gridspec(3, 2)

# bed level
axZb = fig1.add_subplot(gs1[0, :])
axQs = fig1.add_subplot(gs1[1, 0])
axQb = fig1.add_subplot(gs1[1, 1])
axDiff = fig1.add_subplot(gs1[2, 0])
axErr = fig1.add_subplot(gs1[2, 1])

for i in range(ntimes-1):
    t0, t = timeList[i], timeList[i+1]
    dt = round(float(t) - float(t0), 5)
    Xb, Yb, Zb = rdf.readmesh("./", t, boundary="bed")
    dX = Xb[1] - Xb[0]  # mesh is uniform in x-direction

    # saturated bedload
    qsX, qsY, qsZ = rdf.readvector(
        "./", t, "qsatVf", boundary="bed")

    # current time bedload qb, induced by shear stress
    qbX, qbY, qbZ = rdf.readvector(
        "./", t, "qbVf", boundary="bed")

    # previous time bedload qb, induced by shear stress
    qb0X, qb0Y, qb0Z = rdf.readvector(
        "./", t0, "qbVf", boundary="bed")

    qbE = np.zeros(len(Xb) + 1)
    # old qb on edges, linear interpolation on uniform mesh
    qbE[1:-1] = 0.5 * (qb0X[1:] + qb0X[:-1])
    qbE[0], qbE[-1] = qb0X[0], qb0X[-1]
    eFqb = qbE
    # compute current bedload using saturation equation, x-component
    qbPY = qb0X + (dt / Tsat) * (
        qsX - qb0X + (Lsat / dX) * (eFqb[:-1] - eFqb[1:]))

    errQb = (qbX - qbPY) / np.max(np.abs(qbPY))
    maxRelErrQb = np.max(np.abs(errQb))
    print(f"maximum relative error on bedload: {maxRelErrQb}")
    axErr.plot(Xb, errQb, color=colors[i])

    axQb.plot(Xb, qbX, ls="solid", color=colors[i])
    axQb.plot(Xb, qbPY, ls="dashed", color=colors[i])

    # avalanche related bedload flux
    qavX, qavY, qavZ = rdf.readvector(
        "./", t, "qavVf", boundary="bed", verbose=False)

    axZb.plot(Xb, Zb, color=colors[i])
    axQs.plot(Xb, qsX, color=colors[i])
    qDiff = qsX - qbX
    axDiff.plot(Xb, qDiff, color=colors[i])

axZb.set_ylabel(r"$z_b$")

axQs.set_ylabel(r"$q_{sat}$")
axQs.set_xticklabels([])

custom_lines = [Line2D([0], [0], color="black", ls="solid"),
                Line2D([0], [0], color="black", ls="dashed")]
axQb.legend(custom_lines, [r"$q_b^{of}$", r"$q_b^{py}$"])
axQb.set_ylabel(r"$q_b$")
axQb.set_xticklabels([])

axDiff.set_ylabel(r"$q_{sat} - q_b$")
axDiff.set_xlabel(r"$x$")

axErr.set_ylabel(r"$q_b$ relative error")
axErr.set_xlabel(r"$x$")

for ax in fig1.axes:
    ax.set_xlim(-0.05, 0.05)
    ax.grid()

fig1.tight_layout()

plt.show()
