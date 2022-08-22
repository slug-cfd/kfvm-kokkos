#ifndef BOUNDARYCONDITIONS_H_
#define BOUNDARYCONDITIONS_H_

namespace KFVM {

  namespace BoundaryConditions {
    // Work tags for different BC types
    enum BCType {periodic = 0,outflow,reflecting,user};
  }
}

#endif
