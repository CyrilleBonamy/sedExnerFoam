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

#include "projectedFaMesh.H"

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

namespace Foam
{

// vector a is projected on a plane with vector b normal to the plane
vector projectHPlane
(
    const vector& a,
    const vector& planeNormal
)
{
    vector aProj = a - (a & planeNormal) * (planeNormal / mag(planeNormal));
    return aProj;
}

}  // End of namespace Foam


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

void Foam::projectedFaMesh::calcLe() const
{
    if (LePtr_)
    {
        FatalError
            << "LePtr_ already allocated" << endl;
        Info << abort(FatalError);
    }

    LePtr_ = new vectorField(nEdges_);

    const vectorField& Le = mesh_.Le();

    const edgeList& edges = mesh_.edges();

    const pointField& points = mesh_.points();

    vectorField& LeProj = *LePtr_;

    // internal edges
    forAll(Le, edgei)
    {
        const edge& e = edges[edgei];
        // normal vector oriented from owner to neighbour face
        vector edgeNormal =
            projectHPlane(Le[edgei], projectNormal_).normalise();
        // vertex delimiting the edge
        const point& x1 = points[e.start()];
        const point& x2 = points[e.end()];
        // length of projected edge on horizontal plane
        scalar le = mag(projectHPlane(x1 - x2, projectNormal_));
        LeProj[edgei] = le * edgeNormal;
    }
    // boundary edges
    forAll(mesh_.boundary(), patchi)
    {
        const faePatchField<Vector<scalar>>& LeBound =
            mesh_.Le().boundaryField()[patchi];
        label edgei = mesh_.boundary()[patchi].start();
        forAll(mesh_.boundary()[patchi], patchEdgei)
        {
            const edge& e = edges[edgei];
            vector edgeNormal = projectHPlane(
                LeBound[patchEdgei], projectNormal_).normalise();
            // vertex delimiting the edge
            const point& x1 = points[e.start()];
            const point& x2 = points[e.end()];
            scalar le = mag(projectHPlane(x1 - x2, projectNormal_));
            LeProj[edgei] = le * edgeNormal;
            edgei++;
        }
    }
}

void Foam::projectedFaMesh::calcMagLe() const
{
    if (magLePtr_)
    {
        FatalError
            << "LePtr_ already allocated" << endl;
        Info << abort(FatalError);
    }

    magLePtr_ = new scalarField(nEdges_);
    // reference to edge length vector
    const vectorField& Le = this->Le();
    
    scalarField& magLe = *magLePtr_;
    
    forAll(Le, edgei)
    {
        magLe[edgei] = Foam::mag(Le[edgei]);
    }
}

void Foam::projectedFaMesh::calcAreaCentres() const
{
    if (areaCentresPtr_)
    {
        FatalError
            << "faceCentresPtr_ already allocated" << endl;
        Info << abort(FatalError);
    }
    
    areaCentresPtr_ = new vectorField(nFaces_);
    // face centers coordinates on curved faMesh
    const vectorField& areaCentres =
        mesh_.areaCentres().internalField();

    vectorField& areaCentresProj = *areaCentresPtr_;

    forAll(areaCentres, facei)
    {
        areaCentresProj[facei] =
            projectHPlane(areaCentres[facei], projectNormal_);
    }
}

void Foam::projectedFaMesh::calcEdgeCentres() const
{
    if (edgeCentresPtr_)
    {
        FatalError
            << "edgeCentresPtr_ already allocated" << endl;
        Info << abort(FatalError);
    }

    edgeCentresPtr_ = new vectorField(nEdges_);
    // face centers coordinates on curved faMesh
    const vectorField& edgeCentres =
        mesh_.edgeCentres().internalField();

    vectorField& edgeCentresProj = *edgeCentresPtr_;

    forAll(edgeCentres, edgei)
    {
        edgeCentresProj[edgei] =
            projectHPlane(edgeCentres[edgei], projectNormal_);
    }
}

void Foam::projectedFaMesh::calcPointCoords() const
{
    if (pointCoordsPtr_)
    {
        FatalError
            << "pointCoordsPtr_ already allocated" << endl;
        Info << abort(FatalError);
    }

    pointCoordsPtr_ = new vectorField(nPoints_);

    vectorField& pointCoords = *pointCoordsPtr_;
    // faMesh points
    const pointField& faPoints = mesh_.points();

    forAll(faPoints, pointi)
    {
        const point& point = faPoints[pointi];

        pointCoords[pointi] =
            projectHPlane(point, projectNormal_);
    }
}

void Foam::projectedFaMesh::calcS() const
{
    if (SPtr_)
    {
        FatalError
            << "edgeCentresPtr_ already allocated" << endl;
        Info << abort(FatalError);
    }
    SPtr_ = new scalarField(nFaces_);

    const scalarField& S = mesh_.S();
    const vectorField& faceNormals = mesh_.faceAreaNormals();
    scalarField& Sproj = *SPtr_;

    forAll(S, facei)
    {
        Sproj[facei] =
            S[facei] * (faceNormals[facei] & projectNormal_);
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
