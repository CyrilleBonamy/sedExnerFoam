# SedExnerFoam

SedExnerFoam is an OpenFOAM solver in development aiming at the numerical study of various sediment transport and morphodynamics problems.


## Repository organization

The repository is organized as follows:
- **solver**, definition of the solver
- **src**, various librairies needed by the solver (bedload model, settling model, etc.)
- **tests**, cases to help identify bugs and missbehaviors of model 
- **tutorials**, examples of sedExnerFoam applications. cases are categorized as RAS or laminar. A PY folder contains various python scripts and jupyter-notebook to post process simulations and compare results with experimental data contained in the DATA folder.


## Installation

SedExnerFoam can be compiled with version *v2412* of [OpenFoam](https://www.openfoam.com/).

In the repository where *sedExnerFoam* code should be stored:

```bash
git clone --recurse-submodules 'https://github.com/Renaud-Matthias/sedExnerFoam/tree/main'
cd sedExnerFoam
./Allwmake
```


## Model description

The model was developed from the solver pimpleFoam. The hydrodynamics is solved using the *PIMPLE* algorithm for incompressible fluid. Some sediments are transported in suspension, they are advected by the fluid, settle under the influence of gravity and are mixed by turbulence. A erosion/deposition term controls the exchange between the suspended load and the sediment bed. The rest of the sediments are transported as bedload, they slide, roll and saltate on the bed. These different sediment fluxes lead to the morphological evolution described by the Exner equation and handled by the Arbitrary Lagrangian Eulerian approach. With this approach, the sediment bed act as a boundary of the computational domain. The mesh is dynamic and get distorted in time so that the bed boundary match the geometry changes.

- hydrodynamics:

$$ \frac{\partial \mathbf{u}}{\partial t} + \nabla.(\mathbf{u}^T\mathbf{u}) = -\frac{1}{\rho_f}\nabla p + \mathbf{g} + \nabla.(2\nu \mathbf{S} + \mathbf{\tau_f}) $$

$$ \nabla.\mathbf{u}=0 $$

- suspended load transport:

$$ \frac{\partial c_s}{\partial t} + \nabla.[(\mathbf{u}+\mathbf{w_s})c_s] = \nabla.(\epsilon_s \nabla c_s) $$

- bedload transport and morphodynamics are described by the Exner equation:

$$ (1-\lambda_s)\frac{\partial z_b}{\partial t} + \nabla.\mathbf{q_b}= D-E $$

The Exner equation is solved on a finite area mesh, which is a mesh made of finite polyhedral surfaces which can be curved in the 3D space.


## Developers

- Matthias Renaud
- Cyrille Bonamy
- Julien Chauchat
- Olivier Bertrand


## Acknowledgements

[OpenFoam](https://www.openfoam.com/) is the free, open source CFD software developed primarily by OpenCFD Ltd since 2004. The OpenFOAM trademark is owned by OpenCFD Ltd.
