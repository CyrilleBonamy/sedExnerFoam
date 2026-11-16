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

// Access to shear stress induced bedload and avalanche related bedload
areaVectorField& qb = qbPtr.ref();
areaVectorField& qav = qavPtr.ref();

// project qb/qav on horizontal plane
vectorField qbProj = meshProj.project(qb);
vectorField qavProj = meshProj.project(qav);

// projected bedload flux through projected faMesh edges
scalarField& phiqbProj = phiqbProjPtr.ref();

// Initialize to zero (CRITICAL!)
phiqbProj = Zero;

// projected edge centers coordinates
const vectorField& xEdgesProj = meshProj.edgeCentres();
// projected face centers coordinates
const vectorField& xFacesProj = meshProj.areaCentres();
// projected edge vectors, oriented toward owner face
const vectorField& LeProj = meshProj.Le();
// face owner list
const labelList& owner = meshProj.edgeOwner();
// face neighbour list
const labelList& neighbour = meshProj.edgeNeighbour();

// === INTERNAL EDGES ===
if (qbScheme == "centered")
{
    for (label edgei = 0; edgei < meshProj.nInternalEdges(); edgei++)
    {
        scalar distOwner = Foam::mag(xFacesProj[owner[edgei]] -
xEdgesProj[edgei]);
        scalar distNeighbour = Foam::mag(xFacesProj[neighbour[edgei]] -
xEdgesProj[edgei]);
        vector qbEdgei = (distNeighbour * qbProj[owner[edgei]] + distOwner *
qbProj[neighbour[edgei]]) / (distOwner + distNeighbour);

        vector qavEdgei = qavProj[owner[edgei]];
        if (qavScheme == "centered")
        {
            qavEdgei = (distNeighbour * qavProj[owner[edgei]] + distOwner *
qavProj[neighbour[edgei]]) / (distOwner + distNeighbour);
        }
        else if (qavScheme == "upwind")
        {
            if ((qavEdgei & LeProj[edgei]) >= scalar(0.0))
            {
                qavEdgei = qavProj[owner[edgei]];
            }
            else
            {
                qavEdgei = qavProj[neighbour[edgei]];
            }
        }

        phiqbProj[edgei] = (qbEdgei + qavEdgei) & LeProj[edgei];
    }
}
else if (qbScheme == "upwind")
{
    for (label edgei = 0; edgei < meshProj.nInternalEdges(); edgei++)
    {
        vector qbEdgei = qbProj[owner[edgei]];
        if ((qbEdgei & LeProj[edgei]) >= scalar(0.0))
        {
            qbEdgei = qbProj[owner[edgei]];
        }
        else
        {
            qbEdgei = qbProj[neighbour[edgei]];
        }
        vector qavEdgei = qavProj[owner[edgei]];
        if (qavScheme == "centered")
        {
            scalar distOwner = Foam::mag(xFacesProj[owner[edgei]] -
xEdgesProj[edgei]);
            scalar distNeighbour = Foam::mag(xFacesProj[neighbour[edgei]] -
xEdgesProj[edgei]);
            qavEdgei = (distNeighbour * qavProj[owner[edgei]] + distOwner *
qavProj[neighbour[edgei]]) / (distOwner + distNeighbour);
        }
        else if (qavScheme == "upwind")
        {
            if ((qavEdgei & LeProj[edgei]) >= scalar(0.0))
            {
                qavEdgei = qavProj[owner[edgei]];
            }
            else
            {
                qavEdgei = qavProj[neighbour[edgei]];
            }
        }

        phiqbProj[edgei] = (qbEdgei + qavEdgei) & LeProj[edgei];
    }
}

// === BOUNDARY EDGES (CRITICAL FOR MASS CONSERVATION!) ===
// make sure ghost (neighbour processor) values are up to date
qav.correctBoundaryConditions();

forAll(aMesh.boundary(), patchi)
{
    const faPatch& patch = aMesh.boundary()[patchi];

    if (patch.type() == "empty")
    {
        continue;
    }

    const bool coupled = patch.coupled();
    const faPatchField<vector>& QbBound = qb.boundaryField()[patchi];
    const faPatchField<vector>& QavBound = qav.boundaryField()[patchi];

    label edgei = patch.start();
    forAll(patch, patchEdgei)
    {
        const vector edgeLeProj =
            meshProj.project(aMesh.Le().boundaryField()[patchi][patchEdgei]);

        const vector qbOwn = meshProj.project(qb[owner[edgei]]);
        const vector qbGho = meshProj.project(QbBound[patchEdgei]);
        const vector qavOwn = meshProj.project(qav[owner[edgei]]);
        const vector qavGho = meshProj.project(QavBound[patchEdgei]);

        vector qbEdgei;
        vector qavEdgei;

        if (coupled)
        {
            // processor edge: apply the same scheme as on internal edges,
            // using the ghost value as the neighbour face, so that both
            // processors compute the same physical flux (Le has opposite
            // signs on the two sides, giving equal and opposite
            // contributions in the dVsed balance, hence conservative)
            if (qbScheme == "centered")
            {
                qbEdgei = 0.5*(qbOwn + qbGho);
            }
            else  // upwind, sign taken from the centered flux
            {
                const vector qbC = 0.5*(qbOwn + qbGho);
                qbEdgei =
                    ((qbC & edgeLeProj) >= scalar(0.0)) ? qbOwn : qbGho;
            }
            if (qavScheme == "centered")
            {
                qavEdgei = 0.5*(qavOwn + qavGho);
            }
            else
            {
                const vector qavC = 0.5*(qavOwn + qavGho);
                qavEdgei =
                    ((qavC & edgeLeProj) >= scalar(0.0)) ? qavOwn : qavGho;
            }
        }
        else
        {
            // physical boundary: prescribed boundary value
            qbEdgei = qbGho;
            qavEdgei = qavGho;
            if (qavScheme == "upwind")
            {
                qavEdgei = ((qavEdgei & edgeLeProj) >= scalar(0.0))
                    ? qavGho : qavOwn;
            }
        }

        phiqbProj[edgei] = (qbEdgei + qavEdgei) & edgeLeProj;
        edgei++;
    }
}

// === APPLY RIGID BED CONSTRAINT (careful with threshold!) ===
if (bed.rigidBed())
{
    areaScalarField& HsedBed = HsedBedPtr.ref();
    for (label edgei = 0; edgei < meshProj.nEdges(); edgei++)
    {
        if (phiqbProj[edgei] > 0)  // Flux sortant
        {
            if (HsedBed[owner[edgei]] <= 1e-6)
            {
                phiqbProj[edgei] = 0;
            }
        }
        else if
        (
            edgei < meshProj.nInternalEdges()
         && phiqbProj[edgei] < 0
        )  // Flux entrant from an internal neighbour
        {
            if (HsedBed[neighbour[edgei]] <= 1e-6)
            {
                phiqbProj[edgei] = 0;
            }
        }
    }
}

// === CONSERVATIVE UPDATE OF dVsed ===
// Internal edges: flux from owner to neighbour
for (label edgei = 0; edgei < meshProj.nInternalEdges(); edgei++)
{
    dVsed[owner[edgei]] -= phiqbProj[edgei];
    dVsed[neighbour[edgei]] += phiqbProj[edgei];
}

// Boundary edges: flux out of domain (owner is always the internal face)
for (label edgei = meshProj.nInternalEdges(); edgei < meshProj.nEdges();
edgei++)
{
    dVsed[owner[edgei]] -= phiqbProj[edgei];
}
