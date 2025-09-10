/*---------------------------------------------------------------------------*\
Copyright (C) 2022 Matthias Renaud, Cyrille Bonamy, Julien Chauchat
                   and contributors

License
    This file is part of ScourFOAM.

    ScourFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    ScourFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with ScourFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

// access to saturated bedload and bedload fields
areaVectorField& qsat = qsatPtr.ref();
areaVectorField& qb = qbPtr.ref();

dimensionedScalar smallVal(dimless, SMALL);
// unit vector gravity oriented
vector eg = (g/mag(g)).value();

// access criticalShieldsModel
const criticalShieldsModels::criticalShieldsModel& critShieldsMod =
    bed.critShieldsModel();

// access critical Shields number
areaScalarField& critShields = critShieldsPtr.ref();

if (not readCritShields)
{
    // shear stress direction
    vectorField stressDir = shields / (Foam::mag(shields) + SMALL);
    // slope direction
    const vectorField& normalFaces =
        bed.aMesh().faceAreaNormals().internalField();
    // bed faces angle
    scalarField slopeAngle = Foam::acos(normalFaces & eg);
    // slope direction
    vectorField slopeDir = eg - normalFaces * (eg & normalFaces);
    slopeDir /= (Foam::mag(slopeDir) + SMALL);
    //slopeDir /= (Foam::mag(slopeDir) + SMALL);
    // Apply slope correction on critical Shields number
    critShieldsMod.slopeCorrection(
        critShields, stressDir, slopeAngle, slopeDir, betaRep);
}


// compute bedload from Shields number
qsat = bedloadMod.qb(
    shields, critShields, gradZb, rhoS, rhoF, mag(g), dS, nuF);

// apply saturation time to compute bedload
const scalar& deltaT = runTime.time().deltaT().value();
if (Tsat)
{
    //if (Tsat->value() > deltaT)
    //{
    //    qb += (deltaT / Tsat->value()) * (qsat - qb);
    //}

    // access to projected faMesh
    const projectedFaMesh& meshProj = bed.aProjMesh();
    // projected face centers coordinates
    const vectorField& xFacesProj = meshProj.areaCentres();
    // face owner list
    const labelList& owner = meshProj.edgeOwner();
    // face neighbour list
    const labelList& neighbour = meshProj.edgeNeighbour();
    label Nf = meshProj.nFaces();  // number of faces
    Info << "n faces = " << Nf << endl;

    scalarSquareMatrix Amat(meshProj.nFaces());
    scalarSquareMatrix Amatinv(Nf, Zero);
    Amat = Zero;
    scalarField qsx(Nf);
    if (Nf!=0)
    {
        scalar dl = 0;
        for (label edgei = 0; edgei < meshProj.nInternalEdges(); edgei++)
        {
            scalar newdl = Foam::mag(
                xFacesProj[neighbour[edgei]] - xFacesProj[owner[edgei]]);
            if (newdl>dl)
            {
                dl = newdl;
            }
        }
        Info << "dl = " << dl << endl;
        scalar Mcoef = Lsat->value() / (2*dl);
        Info << "Mcoef = " << Mcoef << endl;

        for (label i=1; i<Nf-1; i++)
        {
            Amat[i][i] = 1;
            Amat[i][i+1] = Mcoef;
            Amat[i][i-1] = -Mcoef;
            qsx[i] = qsat[i].x();
        }
        Amat[0][0] = 1 - Mcoef;
        Amat[0][1] = Mcoef;
        Amat[Nf-1][Nf-1] = 1 + Mcoef;
        Amat[Nf-1][Nf-2] = -Mcoef;
        qsx[0] = qsat[0].x();
        qsx[Nf-1] = qsat[Nf-1].x();
    }

    }

    LUscalarMatrix LU(Amat);
    //scalarField qbx(LU.solve(qsx));
    LU.inv(Amatinv);
    scalarField qbx(Amatinv*qsx);

    forAll(meshProj.S(), facei)
    {
        qb[facei].x() = qbx[facei];
    }
}
else
{
    qb = qsat;
}

// additional bedload due to avalanche
if (bedloadMod.avalanche())
{
    #include "sedimAvalanche.H"
}

qb.correctBoundaryConditions();
