#ifndef GEOMETRY_H_
#define GEOMETRY_H_

// File: Geometry.H
// Purpose: Hold grid geometry and convenience functions

#include <Kokkos_Macros.hpp>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {
  
  struct Geometry
  {
    // Bounding box
    const Real xLo,xUp,yLo,yUp,zLo,zUp;
    // Grid spacing
    const Real dx,dy,dz;

    // Construct from Problemsetup object
    Geometry(const ProblemSetup& ps):
      xLo(ps.xLo),xUp(ps.xUp),
      yLo(ps.yLo),yUp(ps.yUp),
      zLo(ps.zLo),zUp(ps.zUp),
      dx((xUp-xLo)/ps.nX),
      dy((yUp-yLo)/ps.nY),
      dz((zUp-zLo)/ps.nZ)
    {}

    // Convenience functions to calculate coordinates at cell centers
    KOKKOS_INLINE_FUNCTION
    Real xCoord(int n) const
    {
      return dx*(Real(n) + 0.5) + xLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real yCoord(int n) const
    {
      return dy*(Real(n) + 0.5) + yLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real zCoord(int n) const
    {
      return dz*(Real(n) + 0.5) + zLo;
    }

  private:
    Geometry() = delete;
  };
}

#endif
