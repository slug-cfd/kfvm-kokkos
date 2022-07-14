#ifndef BOUNDARYCONDITIONS_H_
#define BOUNDARYCONDITIONS_H_

// File: BoundaryConditions.H
// Purpose: Define various boundary condition types

namespace KFVM {

  namespace BoundaryConditions {

    // Work tags for different BC types
    enum BCType {periodic = 0,outflow,reflecting,user};
    struct Periodic {};
    struct Outflow {};
    struct Reflecting {};
    struct User {}; // placeholder, not sure how to implement this one...

  } // end namespace BoundaryConditions

} // end namespace KFVM

#endif

