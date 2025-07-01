"""
Plot bed level at different times.
Plot avalanche bedload flux at different times.
Compute the slope angle at the final tim step with different methods.
Using the bed level, using the bedload flux components.
The bedload flux being tangent to the bed
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from fluidfoam import readof as rdf
import os

plt.rcParams["font.size"] = 15


betaRepDeg = 32.  # repose angle in degrees
betaRep = betaRepDeg * np.pi / 180
Qav0 = 5e-3  # maximum bedload flux related to avalanche

foamTimes = os.popen('foamListTimes').read()
timeList = foamTimes.split('\n')[:-1]
timeArr = np.array([float(t) for t in timeList])

ntimes = len(timeList)

# color list for results
colors = ["cornflowerblue", "tomato"]
cm = LinearSegmentedColormap.from_list(
        "Custom", colors, N=ntimes)
colors = cm(np.arange(0, cm.N))

maxBeta = np.zeros(ntimes)  # beta from solver
maxBetaPY = np.zeros(ntimes)  # beta from qav components
maxBetaZb = np.zeros(ntimes)  # qav from bed level


fig = plt.figure(figsize=(13, 8))

gs = fig.add_gridspec(3, 3)

axZb = fig.add_subplot(gs[0, 0])
axQav = fig.add_subplot(gs[1, 0])
axBeta = fig.add_subplot(gs[2, 0])
axQavErr = fig.add_subplot(gs[1, 1])
axBetaErr = fig.add_subplot(gs[2, 1])
axMaxBeta = fig.add_subplot(gs[:, 2])

for i, time in enumerate(timeList):
    Xbed, Ybed, Zbed = rdf.readmesh(
        "./", time, boundary="bed", verbose=False)
    axZb.plot(Xbed, Zbed, color=colors[i])

    betaOF = rdf.readscalar(
        "./", time, "betaVf", boundary="bed", verbose=False)
    axBeta.plot(Xbed, betaOF*180/np.pi, ls="solid", color=colors[i])
    maxBeta[i] = np.max(betaOF)

    # avalanche bedload flux
    qavX, qavY, qavZ = rdf.readvector(
        "./", time, "qavVf", boundary="bed", verbose=False)
    magQav = np.sqrt(qavX**2 + qavZ**2)
    axQav.plot(Xbed, magQav, color=colors[i])

    qavVinent = Qav0 * (np.tanh(np.tan(betaOF)) - np.tanh(np.tan(betaRep)))
    qavVinent /= (1 - np.tanh(np.tan(betaRep)))

    # relative error on avalanche related bedload
    relErrQav = np.abs(magQav - qavVinent) / Qav0
    axQavErr.plot(Xbed, relErrQav, color=colors[i])

    # compute beta from qav
    gradZb = np.arctan(qavZ/qavX)
    betaPY = np.abs(gradZb)
    axBeta.plot(Xbed, betaPY*180/np.pi, ls="dashed", color=colors[i])
    maxBetaPY[i] = np.max(betaPY)

    relErrBeta = (betaPY - betaOF) / betaOF
    axBetaErr.plot(Xbed, relErrBeta, color=colors[i])

    # compute beta from bed level
    maxBetaZb[i] = np.max(
        np.arctan(
            np.abs((Zbed[1:] - Zbed[:-1])/(Xbed[1:] - Xbed[:-1]))
        )
    )


axZb.set_ylabel(r"$z_b\,[m]$")
axZb.set_xticklabels([])
axZb.grid()

axQav.set_ylabel(r"$q_{av}\,[m^2.s^{-1}]$")
axQav.set_xticklabels([])
axQav.grid()

axBeta.set_ylabel(r"$\beta$ in degrees")
axBeta.set_xlabel("x [m]")
axBeta.grid()

axQavErr.set_ylabel(r"relative error on $q_{av}$")
axQavErr.set_xlabel("x [m]")
axQavErr.grid()

axBetaErr.set_ylabel(r"relative error on $\beta$")
axBetaErr.set_xlabel("x [m]")
axBetaErr.grid()

axMaxBeta.plot(
    timeArr, maxBeta * 180 / np.pi, ls="solid",
    color="#D55E00", label="OpenFOAM")
axMaxBeta.plot(
    timeArr, maxBetaPY * 180 / np.pi, ls="dashed",
    color="#0072B2", label=r"from $q_{av}$")
axMaxBeta.plot(
    timeArr, maxBetaZb * 180 / np.pi, ls="dashdot",
    color="#009E73", label=r"from bed level")
axMaxBeta.axhline(betaRepDeg, color="black", ls="dashed", label="repose angle")
axMaxBeta.set_ylabel(r"$max(\beta)$ in degrees")
axMaxBeta.set_xlabel(r"time [s]")
axMaxBeta.legend()
axMaxBeta.grid()


fig.tight_layout()

plt.show()
