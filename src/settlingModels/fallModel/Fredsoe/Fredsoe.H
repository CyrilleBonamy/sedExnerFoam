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
#ifndef Fredsoe_H
#define Fredsoe_H

#include "fallModel.H"

namespace Foam
{
namespace settlingModels
{

class Fredsoe
:
    public fallModel
{

    protected:

    // Protected member functions

    //- Get settling velocity of a lone particle in a fluid
    dimensionedScalar Ufall0
    (
        const dimensionedScalar& dS,
        const dimensionedScalar& rhoS,
        const dimensionedScalar& rhoF,
        const dimensionedScalar& nuF,
        const dimensionedScalar& g
    ) const;

    //- Solve the Fredsoe model to compute th fall velocity
    dimensionedScalar solveUfall0
    (
        const dimensionedScalar& dS,
        const dimensionedScalar& s,
        const dimensionedScalar& nuF,
        const dimensionedScalar& g
    ) const;

    //- Next value in search for solution
    dimensionedScalar nextValue
    (
        const dimensionedScalar& WsOld,
        const dimensionedScalar& dS,
        const dimensionedScalar& s,
        const dimensionedScalar& nuF,
        const dimensionedScalar& g
    ) const;

public:

    //- Runtime type information
    TypeName("Fredsoe");


    // Constructors

    //- Construct from components
    Fredsoe(const dictionary& dict);


    //- Destructor
    virtual ~Fredsoe();

};

} // End namespace settlingModels
} // End namespace Foam

#endif
