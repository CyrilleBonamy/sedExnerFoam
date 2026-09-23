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

Description
    Bed motion time scale, mass-conservative alternative to clipping dH.

    Uses dH computed at the end of the previous time step (already
    relaxed/converged if nOuterCorrectors > 1) to bound the next deltaT so
    that the bed displacement stays under maxCoBed times the height of the
    first fluid cell above the bed, on every bed face. Purely reactive, like
    CourantNo.H: does not protect the step that produced a too-large dH,
    only the following ones.

    Must be included after CourantNo.H and before setDeltaT.H, in the
    non-LTS branch only.

\*---------------------------------------------------------------------------*/

scalar deltaTBed = GREAT;

if (bed.exist() && bed.bedMotion())
{
    const areaScalarField& dH = dHptr();
    const faMesh& aMesh = bed.aMesh();

    // deltaT of the step that produced dH (current deltaT, not yet updated)
    const scalar deltaT0 = runTime.deltaTValue();

    scalar minRatio = GREAT;

    for (const label patchID : bed.bedPatchesID())
    {
        const fvPatch& bedPatch = mesh.boundary()[patchID];

        forAll(bedPatch, facei)
        {
            const label idNCell = bedPatch.faceCells()[facei];
            const vector& xCell = mesh.C()[idNCell];
            const vector& xFace = aMesh.areaCentres()[facei];
            const vector& nFace = aMesh.faceAreaNormals()[facei];

            // height of the first fluid cell above the bed face
            const scalar hLocal = -(xCell - xFace) & nFace;

            const scalar ratio = hLocal / (mag(dH[facei]) + SMALL);
            minRatio = min(minRatio, ratio);
        }
    }

    reduce(minRatio, minOp<scalar>());

    deltaTBed = maxCoBed * minRatio * deltaT0;

    Info << "Bed motion time scale: max deltaT = " << deltaTBed << " s"
        << endl;
}
