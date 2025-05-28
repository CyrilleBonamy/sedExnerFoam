"""
Plot streamwise velocity, diffusivity for suspended sediments
turbulent kinetic energy and the volume fraction of suspended sediments
"""

import numpy as np
import matplotlib.pyplot as plt
from fluidfoam import readof as rdf
import os

plt.rcParams["font.size"] = 15

time = "50"

# simulation parameters
Hwater = 0.1  # water depth
dS = 0.2e-3  # sand particle diameter (m)
nuF = 1e-6  # fluid kinematic viscosity
rhoS = 2650.  # sediment density
rhoF = 1000.  # water density
g = 9.81  # gravity acceleration
Sc = 1.  # Schmidt number
kappa = 0.4  # von Karman constant
ks = 2.5 * dS  # Nikuradse equivalent roughness


def getUmean(U, Z):
    """"""
    dz0 = 2*Z[0]
    dZ = [dz0]
    for i in range(len(Z)-1):
        dz = (Z[i+1] - Z[i] - 0.5*dZ[i]) * 2
        dZ.append(dz)
    Umean = sum([u*dz for u, dz in zip(U, dZ)])/Hwater
    return Umean


def RouseProfile(z, z0, c0, Ro):
    """Rouse equilibrium profile for suspended sediments

    Parameters
        z: float
            distance from wall
        z0: float
            reference level close to the bottom
        c0: float
            concentration at level z0
        Ro: float
            Rouse number
    """
    Crouse = ((z0 / (Hwater - z0)) * ((Hwater - z)/z))**Ro
    return c0 * Crouse


# load results from simulation
Zmesh = rdf.readmesh("./")[2]
# x component of velocity field
UxField = rdf.readfield("./", time, "U")[0]
# turbulent kinetic energy
kField = rdf.readscalar("./", time, "k")
omegaField = rdf.readscalar("./", time, "omega")
nutField = rdf.readscalar("./", time, "nut")
diffSed = rdf.readscalar("./", time, "diffSed")
CsField = rdf.readscalar("./", time, "Cs")

# get wall shear stress and friction velocity, m2/s2
tauOf = rdf.readvector(
    "./", time, "wallShearStress", boundary="bed", verbose=False)[0, 0]
shields = rdf.readvector(
    "./", time, "shieldsVf", boundary="bed", verbose=False)[0, 0]
shieldsOf = -tauOf / ((rhoS / rhoF - 1) * g * dS)
ufOf = np.sqrt(-tauOf)
ufSolv = np.sqrt(shields * (rhoS/rhoF - 1) * g * dS)

dVisc = 12 * nuF / ufSolv  # viscous sublayer thickness

print(f"Shields number from wallShearStress: {shieldsOf} m/s")
print(f"Shields number from sedExnerFoam: {shields} m/s")
print(f"friction velocity from wallShearStress: {ufOf} m/s")
print(f"friction velocity from sedExnerFoam: {ufSolv} m/s")
print(f"\nShields number = {shields}")
kPlus = ks * ufSolv / nuF  # dimension less roughness length
print(f"bed equivalent roughness height, ks = {ks} m")
print(f"dimension less roughness, k+ = {kPlus}")

print(f"\nz+ (z * uf / nu) = {Zmesh[0]*ufOf/nuF}")

print("\nsimulation mean velocity: ", getUmean(UxField, Zmesh), "m/s")

# load results from prevous simulation
zData, CsData = np.loadtxt(
    "./dataCs_50s.txt", unpack=True, delimiter=";")
# relative error on Cs profile
errCs = np.abs(CsField - CsData) / CsData


fig, (axU, axNut, axK, axCs, axErr) = plt.subplots(
    ncols=5, figsize=(16, 6))

axU.plot(UxField, Zmesh/Hwater, lw=2, color="steelblue")
axU.set_xlabel("U [m/s]")
axU.set_ylabel("z/H")
axU.grid()

axNut.plot(nutField, Zmesh/Hwater, lw=2, color="steelblue", label=r"$\nu_t$")
axNut.plot(
    diffSed, Zmesh/Hwater, lw=2, ls="dashed", color="firebrick",
    label=r"$\nu_t/\sigma_c + \epsilon_{wall}$")
axNut.axhline(ks/Hwater, ls="dashed", color="black", label=r"$k_s$")
axNut.axhline(dVisc/Hwater, ls="dashed", color="black", label=r"$\delta_v$")
axNut.set_xlabel(r"diffusivity $[m^2.s^{-1}$")
axNut.legend()
axNut.grid()

axK.plot(kField, Zmesh/Hwater, lw=2, color="steelblue")
axK.set_xlabel(r"$k\,[m^2.s^{-2}]$")
axK.grid()

axCs.scatter(CsField, Zmesh/Hwater, lw=2, color="steelblue")
axCs.scatter(CsData, zData/Hwater, marker="x", color="black")
axCs.set_xlabel(r"$c_s$")
axCs.set_xscale("log")
axCs.grid()

axErr.plot(errCs, Zmesh/Hwater, marker="x", color="steelblue")
axErr.set_xlabel(r"$c_s$ relative error")
axErr.grid()

for ax in fig.axes[1:]:
    ax.set_yticklabels([])

fig.tight_layout()

plt.show()
