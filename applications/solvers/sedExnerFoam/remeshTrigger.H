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
    Detection-only, no action yet: flags when the cumulated bed level
    increment since the last remesh becomes large compared to the local
    first fluid cell height, over bed.bedPatchesID(). Generic on purpose:
    works for a flat/dune bed as-is, and extends to a pile/tripod case by
    also checking additional obstacle patches once mesh-quality indicators
    (orthogonality, skewness) are added in a later step.

    Included once per time step, after exnerEqn.H (dH is available and
    up to date). Does not reset dHCumul or trigger any actual remeshing:
    for now this is a calibration tool to choose remeshCoBed and to see,
    on a real run, when and where a remesh would be requested.

\*---------------------------------------------------------------------------*/

// dH is the increment over the whole time step, recomputed (and relaxed)
// at every PIMPLE outer sub-iteration: only accumulate once it is
// converged, at the final outer iteration, otherwise the same time step
// would be counted several times in dHCumul.
if (bed.exist() && bed.bedMotion() && pimple.finalIter())
{
    areaScalarField& dHCumul = dHCumulPtr.ref();
    dHCumul.primitiveFieldRef() += dH.primitiveFieldRef();

    const faMesh& aMesh = bed.aMesh();

    scalar ratioMin = GREAT;
    label patchAtMin = -1;
    label faceAtMin = -1;

    for (const label patchID : bed.bedPatchesID())
    {
        const fvPatch& bedPatch = mesh.boundary()[patchID];

        forAll(bedPatch, facei)
        {
            const label idNCell = bedPatch.faceCells()[facei];
            const vector& xCell = mesh.C()[idNCell];
            const vector& xFace = aMesh.areaCentres()[facei];
            const vector& nFace = aMesh.faceAreaNormals()[facei];

            const scalar hLocal = -(xCell - xFace) & nFace;
            const scalar ratio =
                hLocal / (mag(dHCumul[facei]) + SMALL);

            if (ratio < ratioMin)
            {
                ratioMin = ratio;
                patchAtMin = patchID;
                faceAtMin = facei;
            }
        }
    }

    // parallel reduction: keep min value only (processor/face location
    // is lost across processors, fine for calibration purposes)
    reduce(ratioMin, minOp<scalar>());

    Info << "Remesh criterion (bed displacement): min ratio = "
        << ratioMin << " (threshold = " << remeshCoBed << ")" << endl;

    if (ratioMin < remeshCoBed)
    {
        Info << "*** Remesh would be required now (bed displacement), "
            << "t = " << runTime.timeName()
            << ", min ratio = " << ratioMin;
        if (!Pstream::parRun() && patchAtMin != -1)
        {
            Info << ", on patch " << mesh.boundary()[patchAtMin].name()
                << ", face " << faceAtMin;
        }
        Info << " ***" << endl;
    }
}
