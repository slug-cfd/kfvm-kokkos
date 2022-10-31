#ifndef SIMVAR_H_
#define SIMVAR_H_

// File: SimVar.H
// Usage: This file must contain defined enums for the simulation variables
//        evolved by the solver.
// Requirements: This must define "enum SimVar { ... }", and the last
//               entry must be NUM_VARS.

#include<cstddef>
#include<string>

enum SimVar { dens = 0, momx, momy, momz, etot, NUM_VARS };

enum AuxVar { velx = 0, vely, velz, eint, pres, NUM_AUX };

enum FluidProp { gam = 0, NUM_PROPS };

#endif
