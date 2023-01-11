#ifndef GEOMETRY_H_
#define GEOMETRY_H_

// File: Geometry.H
// Purpose: Hold grid geometry and convenience functions

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {
  
  struct Geometry
  {
    // Bounding box
    const Real xLo,xUp,yLo,yUp,zLo,zUp;
    // Grid spacing
    const Real dx,dy,dz,dmin;

    // Construct from Problemsetup object
    Geometry(const ProblemSetup& ps):
      xLo(ps.xLo),xUp(ps.xUp),
      yLo(ps.yLo),yUp(ps.yUp),
      zLo(ps.zLo),zUp(ps.zUp),
      dx((xUp-xLo)/ps.nX),
      dy((yUp-yLo)/ps.nY),
      dz((zUp-zLo)/ps.nZ),
#if (SPACE_DIM == 2)
      dmin(std::fmin(dx,dy))
#else
      dmin(std::fmin(dx,std::fmin(dy,dz)))
#endif
    {}

    // Convenience functions to calculate coordinates at cell centers
    KOKKOS_INLINE_FUNCTION
    Real xCoord(idx_t n) const
    {
      return dx*(Real(n) + 0.5) + xLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real yCoord(idx_t n) const
    {
      return dy*(Real(n) + 0.5) + yLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real zCoord(idx_t n) const
    {
      return dz*(Real(n) + 0.5) + zLo;
    }

  private:
    Geometry() = delete;
  };
}

#endif
