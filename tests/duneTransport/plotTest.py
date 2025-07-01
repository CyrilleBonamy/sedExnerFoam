"""
Plot bed level from simulation and compare with previous results
Plot Shields number, critical Shields number
Plot bedload flux and compare values with empirical formulas
Plot slope angle with different methods and compare them
"""

import numpy as np
from fluidfoam import readof as rdf
import matplotlib.pyplot as plt
import os

plt.rcParams["font.size"] = 15

# physical parameters
dS = 0.4e-3  # sediment diameter
rhoS = 2500.  # sediment density
rhoF = 1000.  # fluid density
nuF = 1e-6  # fluid kinematic viscosity
g = 9.81  # gravity acceleration
betaRepDeg = 28.  # granular material repose angle in degrees
betaRep = betaRepDeg * np.pi / 180

einNum = np.sqrt((rhoS/rhoF - 1) * g * dS**3)
print(f"Einstein number, Ei = {einNum} m2/s")

Dstar = dS * ((rhoS/rhoF - 1)*g / nuF**2)**(1/3)
print(f"dimless diameter, Dstar = {Dstar}")

critShSoulsby = (0.3 / (1 + 1.2*Dstar)) + 0.055*(1 - np.exp(-0.02*Dstar))
print(f"critical Shields number, {critShSoulsby}")

foamTimes = os.popen('foamListTimes').read()
timeList = foamTimes.split('\n')[:-1]
print("time list: ", timeList)
ntimes = len(timeList)

time = "latestTime"

Xb, Yb, Zb = rdf.readmesh("./", time, boundary="bed")
# load xb/zb values from reference simulation
XbPrev, ZbPrev = np.loadtxt(
    "./dataZb01.txt", unpack="True", delimiter=";")
relErrZb = np.abs(Zb - ZbPrev) / np.max(ZbPrev)
print(f"max error on bed level: {np.max(relErrZb)}")

# critical Shields number with slope correction
critShields = rdf.readscalar(
    "./", time, "critShieldsVf", boundary="bed")
if critShields.shape == (1,):
    critShields = np.ones_like(Xb) * critShields

# Shields number, bed shear stress
shX, shY, shZ = rdf.readvector(
    "./", time, "shieldsVf", boundary="bed")
magSh = np.sqrt(shX**2 + shZ**2)

# saturated bedload
qsX, qsY, qsZ = rdf.readvector(
    "./", time, "qsatVf", boundary="bed")
magQs = np.sqrt(qsX**2 + qsZ**2)

# bedload qb, induced by shear stress
qbX, qbY, qbZ = rdf.readvector(
    "./", time, "qbVf", boundary="bed")
magQb = np.sqrt(qbX**2 + qbZ**2)

# avalanche related bedload flux
qavX, qavY, qavZ = rdf.readvector(
    "./", time, "qavVf", boundary="bed")
magQav = np.sqrt(qavX**2 + qavZ**2)
# mask, X position where avalanche occur
Xmask = np.ma.masked_where(np.abs(qavX) < 1e-10, Xb)
qavXmask = np.ma.masked_where(np.abs(qavX) < 1e-10, qavX)
qavZmask = np.ma.masked_where(np.abs(qavX) < 1e-10, qavZ)

# beta from solver, positive values
betaOF = rdf.readscalar(
    "./", time, "betaVf", boundary="bed")

# compute slope angle beta from Shields components
betaSh = np.arctan(shZ / shX)
betaErrSh = np.abs(
    (np.abs(betaSh) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))

# compute beta from qav components, masked array
betaQav = np.arctan(qavZmask / qavXmask)
betaErrQav = np.abs(
    (np.abs(betaQav) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))

# compute beta from qb components
betaQb = np.arctan(qbZ / qbX)
betaErrQb = np.abs(
    (np.abs(betaQb) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))

# compute beta from qsat components
betaQsat = np.arctan(qsZ / qsX)
betaErrQsat = np.abs(
    (np.abs(betaQsat) - np.abs(betaOF)) / (np.abs(betaOF) + 1e-10))

# critical Shields slope correction
cosAlpha = np.where(shX*betaSh > 0, -1, 1)
betaEff = np.where(np.abs(betaSh) < betaRep, np.abs(betaSh), betaRep)
slopeCorr = np.cos(betaEff) - cosAlpha * np.sin(betaEff) / np.tan(betaRep)
critShCorr = critShSoulsby * slopeCorr

# relative error on critical Shields number
# with slope Correction
errCritShields = np.abs(critShields - critShCorr) / np.max(critShCorr)
print(f"maximum relative error on critShields, {np.max(errCritShields)}")

# bedload from Meyer-Peter & Müller formula
qbMPM = einNum * 8 * np.where(
    magSh > critShields, magSh - critShields, 0)**1.5
# error on saturated bedload flux, comparison with Meyer-Peter
relErrQs = np.abs(magQs - qbMPM) / np.max(qbMPM)
print(
    f"maximum relative error on saturated bedload, {np.max(relErrQs)}")
# bedload due to avalanche from Vinent et. al (2019) formula
Qav0 = 5e-3  # maximum magnitude of avalanche
qavPY = Qav0 * np.where(
    np.abs(betaOF) > betaRep,
    (np.tanh(np.tan(np.abs(betaOF)))-np.tanh(np.tan(betaRep)))
    / (1-np.tanh(np.tan(betaRep))), 0)
relErrQav = np.abs(magQav - qavPY) / np.max(qavPY)


fig1 = plt.figure(figsize=(14, 7))
gs1 = fig1.add_gridspec(2, 2)

fig2 = plt.figure(figsize=(14, 7))
gs2 = fig2.add_gridspec(2, 2)

# bed level
axZb = fig1.add_subplot(gs1[0, 0])
axZb.plot(Xb, Zb, color="#0072B2")
axZb.set_ylabel(r"$z_b$")
axZb.set_xlabel("x [m]")
axZb.grid()

# error on bed level
axErrZb = fig1.add_subplot(gs1[1, 0])
axErrZb.plot(
    Xb, relErrZb, lw=2, ls="solid", marker="x", color="#0072B2")
axErrZb.set_ylabel(r"relative error on $z_b$")
axErrZb.grid()

# bed slope angle
axBeta = fig1.add_subplot(gs1[0, 1])
axBeta.plot(
    Xb, betaOF*180/np.pi, ls="solid", lw=2,
    color="#0072B2", label="from solver")
axBeta.plot(
    Xb, np.abs(betaSh*180/np.pi), ls="solid", lw=2,
    color="#D55E00", label=r"from $\theta$")
axBeta.scatter(
    Xmask, np.abs(betaQav*180/np.pi), marker="+",
    color="#009E73", label=r"from $q_{av}$")
axBeta.plot(
    Xb, np.abs(betaQsat*180/np.pi), ls="dashdot", lw=2,
    color="#0072B2", label=r"from $q_{sat}$")
axBeta.plot(
    Xb, np.abs(betaQb*180/np.pi), ls="dotted", lw=3,
    color="#CC79A7", label=r"from $q_b$")
axBeta.axhline(
    betaRepDeg, ls="dashed", color="black", label="repose angle")
axBeta.set_ylabel(r"$\beta$")
axBeta.set_xlabel("x [m]")
axBeta.legend()
axBeta.grid()

# error on bed slope angle comparison with value from solver
axErrBeta = fig1.add_subplot(gs1[1, 1])
axErrBeta.plot(
    Xb, betaErrSh, ls="solid", lw=2,
    color="#D55E00", label=r"from $\theta$")
axErrBeta.scatter(
    Xmask, betaErrQav, marker="+",
    color="#009E73", label=r"from $q_{av}$")
axErrBeta.plot(
    Xb, betaErrQsat, ls="dashdot", lw=2,
    color="#0072B2", label=r"from $q_{sat}$")
axErrBeta.plot(
    Xb, betaErrQb, ls="dotted", lw=3,
    color="#CC79A7", label=r"from $q_b$")
axErrBeta.set_ylabel(r"relative error on $\beta$")
axErrBeta.set_xlabel("x [m]")
axErrBeta.legend()
axErrBeta.grid()

# critical Shields number
axSh = fig2.add_subplot(gs2[0, 0])
axSh.axhline(
    critShSoulsby, ls="dashed", color="black", label=r"$\theta_c^0$")
axSh.plot(
    Xb, critShields, ls="solid", color="#0072B2",
    label=r"$\theta_c^{corr}$ (solver)")
axSh.plot(
    Xb, critShCorr, ls="dashed", color="#D55E00",
    label=r"$\theta_c^{corr}$ (from slope)")
axSh.legend(fontsize=12)
axSh.set_ylabel(r"$\theta_c$")
axSh.grid()

# error on critical Shields number slope correction
axErrSh = fig2.add_subplot(gs2[1, 0])
axErrSh.plot(Xb, errCritShields, marker="x", color="#0072B2")
axErrSh.set_ylabel(r"relative error on $\theta_c$")
axErrSh.grid()

# bedload flux, shear stress inudec and avalanche
axQb = fig2.add_subplot(gs2[0, 1])
axQb.plot(
    Xb, magQs, ls="solid", color="#56B4E9", label=r"$q_{sat}$")
axQb.plot(
    Xb, qbMPM, ls="dashed", color="#0072B2", label="Meyer-Peter")
axQb.scatter(
    Xmask, magQav, marker="+", color="#009E73", label=r"$q_{av}$")
axQb.plot(
    Xb, qavPY, ls="dashed", color="#D55E00", label="Vinent")
axQb.plot(
    Xb, magQb, ls="dotted", lw=3, color="#CC79A7", label=r"$q_b$")
axQb.set_ylabel(r"$q_b\,[m^2.s^{-1}]$")
axQb.legend(fontsize=12)
axQb.grid()

# bedload flux, shear stress inudec and avalanche
axErrQb = fig2.add_subplot(gs2[1, 1])
axErrQb.plot(
    Xb, relErrQs, marker="x", ls="solid",
    color="#009E73", label=r"$q_{sat}$")
axErrQb.plot(
    Xb, relErrQav, marker="x", ls="dashed",
    color="#D55E00", label=r"$q_{av}$")
axErrQb.set_ylabel(r"relative error on $q_b$")
axErrQb.legend(fontsize=12)
axErrQb.grid()


fig1.tight_layout()
fig2.tight_layout()

plt.show()
