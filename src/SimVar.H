#ifndef SIMVAR_H_
#define SIMVAR_H_

// File: SimVar.H
// Usage: This file must contain defined enums for the simulation variables
//        evolved by the solver.
// Requirements: This must define "enum SimVar { ... }", and the last
//               entry must be NUM_VARS.

#include <Definitions.H>

#ifdef KFVM_EQ_MHD

enum SimVar { dens = 0, momx, momy, momz, etot, magx, magy, magz, phi, NUM_VARS };
enum AuxVar { velx = 0, vely, velz, eint, pres, prsg, prst, NUM_AUX };

#else
/* Default to Euler equations */
enum SimVar { dens = 0, momx, momy, momz, etot, NUM_VARS };
enum AuxVar { velx = 0, vely, velz, eint, pres, NUM_AUX };

#endif

#endif
