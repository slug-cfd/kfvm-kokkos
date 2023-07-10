#ifndef DIMENSION_H_
#define DIMENSION_H_

#include <Definitions.H>

// File: Dimension.H
// Usage: This file declares macros useful for dimension dependent
//        code

// Ensure that space dimension is actually set
#ifndef SPACE_DIM
#define SPACE_DIM 2
#warning Defaulting to 2 space dimensions
#endif

// Ensure that space dimension is 2 or 3
#if (SPACE_DIM != 2 && SPACE_DIM != 3)
#error Space dimension must be 2 or 3
#endif

// Set AMReX style dimension dependent declarators
#if (SPACE_DIM == 2)
#define KFVM_D_DECL(a, b, c) a, b
#define KFVM_DM_DECL(a, b) a
#else
#define KFVM_D_DECL(a, b, c) a, b, c
#define KFVM_DM_DECL(a, b) a, b
#endif

// Ensure that number of face quadrature points is defined
// and set total depending on dimension of the faces
#ifndef NUM_QUAD_PTS
#define NUM_QUAD_PTS 3
#warning Defaulting to 3 quadrature points per dimension
#endif

#endif
