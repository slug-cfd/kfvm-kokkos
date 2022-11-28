#ifndef FLUIDPROPERTIES_H_
#define FLUIDPROPERTIES_H_

// File: FluidProperties.H
// Purpose: Hold all fluid properties in an easy to pass-by-value way

#include <Definitions.H>

#include "Types.H"

namespace KFVM {
  
  struct FluidProperties
  {
    Real gamma; // Ratio of specific heats
    Real wenoThresh; // Threshold for weno selector
    Real ch_glm; // Hyperbolic cleaning speed for GLM (not constant!)

    FluidProperties():
      gamma(1.4),
      wenoThresh(10.0),
      ch_glm(0.0)
    {}

    FluidProperties(const Real gamma_,const Real wenoThresh_):
      gamma(gamma_),
      wenoThresh(wenoThresh_),
      ch_glm(0.0)
    {}
  };
}

#endif
