#ifndef FLUIDPROPERTIES_H_
#define FLUIDPROPERTIES_H_

// File: FluidProperties.H
// Purpose: Hold all fluid properties in an easy to pass-by-value way

#include <Definitions.H>

#include "Types.H"

namespace KFVM {
  
  struct FluidProperties
  {
    const Real gamma; // Ratio of specific heats

    Real ch_glm; // Hyperbolic cleaning speed for GLM (not constant!)

    // Construct from Problemsetup object
    FluidProperties(const Real gamma_):
      gamma(gamma_),
      ch_glm(0.0)
    {}

  private:
    FluidProperties() = delete;
  };
}

#endif
