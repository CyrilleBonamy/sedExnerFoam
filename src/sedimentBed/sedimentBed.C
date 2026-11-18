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

#include "sedimentBed.H"

// * * * * * * * * * * * * * * * Local Functions * * * * * * * * * * * * * * //

namespace Foam
{
}

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::sedimentBed::sedimentBed
(
    dictionary& dict,
    Foam::fvMesh& mesh,
    const meshObjects::gravity& g
)
:
    dict_(dict),
    bedExist_(false),
    bedMotion_(false),
    rigidBed_(false),
    mesh_(mesh),
    bedloadModel_(nullptr),
    g_(g),
    eg_((g_ / Foam::mag(g)).value())
{
    checkBedExistence();
    if (bedExist_)
    {
        Info << "create finite-area mesh" << endl;
        aMesh_.reset(new faMesh(mesh_));
        vsm.reset(new volSurfaceMapping(aMesh_));
        getPatchesID();
        // check if bed inclination consistent with gravity
        checkFaMeshOrientation();
        // instantiate projected finite-area mesh
        aProjMesh_.reset(new projectedFaMesh(aMesh_.ref(), eg_));
        // instantiate bedloadModel
        getBedloadModel();
        // instantiate criticalShieldsModel
        getCritShieldsModel();

        // activate/deactivate presence of a rigid bed below sediment layer
        word switchRigidBed =
            dict_.lookupOrDefault<word>("rigidBed", "off");
        if (switchRigidBed=="on")
        {
            rigidBed_ = true;
        }
        
        Info << "faMesh patches ID : " << bedPatchesID_ << endl;
        Info << "faMesh patches names : " << bedPatchesNames_ << endl;
    }
}

// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::sedimentBed::~sedimentBed()
{}

// * * * * * * * * * * * * Public Member Functions * * * * * * * * * * * * * //

const faMesh& Foam::sedimentBed::aMesh()
{
    return aMesh_.ref();
}

const Foam::projectedFaMesh& Foam::sedimentBed::aProjMesh()
{
    return aProjMesh_.ref();
}

const Foam::bedloadModels::bedloadModel&
Foam::sedimentBed::bedloadModel()
{
    return bedloadModel_.ref();
}

Foam::criticalShieldsModels::criticalShieldsModel&
Foam::sedimentBed::critShieldsModel()
{
    return critShieldsModel_.ref();
}

labelList Foam::sedimentBed::bedPatchesID()
{
    return bedPatchesID_.clone();
}

bool Foam::sedimentBed::exist() const
{
    if (bedExist_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Foam::sedimentBed::bedMotion() const
{
    if (bedMotion_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool Foam::sedimentBed::rigidBed() const
{
    if (rigidBed_)
    {
        return true;
    }
    else
    {
        return false;
    }
}

const Foam::scalarField& Foam::sedimentBed::beta() const
{
    if (betaPtr_==nullptr)
    {
        calcBeta();
    }

    return *betaPtr_;
}

const Foam::vectorField& Foam::sedimentBed::slopeDir() const
{
    if (slopeDirPtr_==nullptr)
    {
        calcSlopeDir();
    }

    return *slopeDirPtr_;
}

void Foam::sedimentBed::interpFaceToVertices
(
    scalarField& dHfaces,
    scalarField& dHpoints
)
{
    // Access to face centers in 3D (not the horizontal projection: the
    // sub-triangle weights must not degenerate on steep beds)
    const vectorField& xFaces3D = aMesh_->areaCentres();

    // Access to point coordinates in 3D
    const pointField& xPoints3D = aMesh_->points();

    // normalise interpolation weight
    scalarField normWeight(aMesh_->nPoints());
    normWeight = Zero;

    // loop over all faces
    for (label facei=0; facei < aMesh_->nFaces(); facei++)
    {
        const face& f = aMesh_->faces()[facei];
        const vector& xf = xFaces3D[facei];
        // number of vertices defining face f
        const label nVerts = f.size();

        // loop over points belonging to face
        for (label pi=0; pi < nVerts; pi++)
        {
            // label of point pi, global
            const label piGlob = f.thisLabel(pi);

            scalar areaPF = 0;

            // vertices coordinates
            const vector& xv = xPoints3D[piGlob];
            const vector& xn = xPoints3D[f.nextLabel(pi)];
            const vector& xp = xPoints3D[f.prevLabel(pi)];

            // vector from vertice to face center
            vector vp = xf - xv;
            // vector from vertice to right edge center
            vector vredg = 0.5 * (xn - xv);
            // vector from vertice to left edge center
            vector vledg = 0.5 * (xp - xv);

            areaPF +=
                0.5 * Foam::mag(vredg ^ vp)
                + 0.5 * Foam::mag(vp ^ vledg);

            dHpoints[piGlob] += areaPF * dHfaces[facei];
            normWeight[piGlob] += areaPF;
        }
    }
    if (Pstream::parRun())
    {
        // map faMesh local point labels to global (volume mesh) point labels
        const labelList& globLabels = aMesh_->patch().meshPoints();

        // scatter local contributions into global point labels and sum them
        // across processors, so that points shared by several processors
        // (processor patches) get the total over all their adjacent faces
        scalarField sumH(mesh_.nPoints(), Zero);
        scalarField sumW(mesh_.nPoints(), Zero);
        forAll(dHpoints, pi)
        {
            sumH[globLabels[pi]] += dHpoints[pi];
            sumW[globLabels[pi]] += normWeight[pi];
        }
        reduce(sumH.data(), sumH.size(), sumOp<scalar>(), UPstream::msgType());
        reduce(sumW.data(), sumW.size(), sumOp<scalar>(), UPstream::msgType());
        forAll(dHpoints, pi)
        {
            dHpoints[pi] = sumH[globLabels[pi]];
            normWeight[pi] = sumW[globLabels[pi]];
        }
    }

    // Normalise interpolation weights
    dHpoints /= normWeight;
}

// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

void Foam::sedimentBed::checkBedExistence()
{
    word isBed(dict_.lookup("sedimentBed"));
    if (isBed=="on")
    {
        bedExist_ = true;
        checkBedMotion();
        //checkAvalancheModel_();
    }
    else if (isBed=="off")
    {
        bedExist_ = false;
    }
    else
    {
        FatalError << "wrong keyword for entry sedimentBed!"
            << " possible options are: on off" << endl;
        Info << abort(FatalError) << endl;
    }
}

void Foam::sedimentBed::checkFaMeshOrientation() const
{
    if (not bedExist_)
    {
        return;
    }
    Info << "g : " << g_.value() << endl;
    forAll(aMesh_->faceLabels(), facei)
    {
        double res = g_.value() & aMesh_->faceAreaNormals()[facei];
        if (res <= 0)
        {
            FatalError
                << "sediment bed and gravity orientation "
                << "are not consistent" << endl
                << "g and patch faces dot product "
                << "should be positive" << endl;
            Info << abort(FatalError) << endl;
        }
    }
}

void Foam::sedimentBed::getPatchesID()
{
    List<word> bedPatchesNames(dict_.lookup("sedimentBedPatches"));
    forAll(bedPatchesNames, i)
    {
        word patchName = bedPatchesNames[i];
        label patchID = mesh_.boundaryMesh().findPatchID(patchName);
        if (patchID==-1)
        {
            FatalError << "bedPatch " << patchName
                << " does not exist" << endl
                << "existing patches are:" << endl
                << mesh_.boundaryMesh().names() << endl;
            Info << abort(FatalError) << endl;
        }
        bedPatchesNames_.append(patchName);
        bedPatchesID_.append(patchID);
    }
}

void Foam::sedimentBed::checkBedMotion()
{
    if (dict_.found("bedMotion"))
    {
        word bedMotionState(dict_.lookup("bedMotion"));
        if (bedMotionState=="on")
        {
            bedMotion_ = true;
            Info << "bed motion activated" << endl;
        }
        else if (bedMotionState=="off")
        {
            bedMotion_ = false;
            Info << "no bed motion, "
                << "bedload and shields are computed "
                << "but bed will not move" << endl;
        }
        else
        {
            FatalError << "wrong keyword for entry bedMotion!"
                << " possible options are: on off,"
                << " default is on" << endl;
            Info << abort(FatalError) << endl;
        }
    }
    else
    {
        bedMotion_ = true;
    }
}

void Foam::sedimentBed::getBedloadModel()
{
    // default dict in case bedloadModel not specified
    dictionary defaultBedloadDict;
    defaultBedloadDict.add(keyType("type"), "MeyerPeter");

    if (dict_.found("bedloadModel"))
    {
        const dictionary bedloadModelDict =
            dict_.subDict("bedloadModel");

        bedloadModel_.reset
        (
            bedloadModels::bedloadModel::New
            (
                bedloadModelDict
            )
        );
    }
    else
    {
        bedloadModel_.reset(
            bedloadModels::bedloadModel::New
            (
                defaultBedloadDict
            )
        );
    }
}

void Foam::sedimentBed::getCritShieldsModel()
{
    // default dict in case criticalShieldsModel not specified
    dictionary defaultCritShieldsDict;
    defaultCritShieldsDict.add(keyType("type"), "fixedValue");
    defaultCritShieldsDict.add(keyType("value"), 0.047);
    if (dict_.found("criticalShieldsModel"))
    {
        const dictionary& critShieldsDict =
            dict_.subDict("criticalShieldsModel");
        word modelType(critShieldsDict.lookup("type"));
        if (modelType != "readFromFile")
        {
            critShieldsModel_.reset
                (
                    criticalShieldsModels::criticalShieldsModel::New
                    (
                        critShieldsDict
                    )
                );
        }
    }
    else
    {
        critShieldsModel_.reset
        (
            criticalShieldsModels::criticalShieldsModel::New
            (
                defaultCritShieldsDict
            )
        );
    }
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
