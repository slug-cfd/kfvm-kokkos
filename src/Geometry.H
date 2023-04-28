#ifndef GEOMETRY_H_
#define GEOMETRY_H_

// File: Geometry.H
// Purpose: Hold grid geometry and convenience functions

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {

  // Store local geometry information
  //   e.g. xLo is the lower bound for this rank's x-coordinate
  struct Geometry
  {
    // Grid spacing
    const Real dx,dy,dz,dmin;
    // Bounding box
    const Real xLo,xUp,yLo,yUp,zLo,zUp;

    // Construct from Problemsetup object
    Geometry(const ProblemSetup& ps):
      dx((ps.xUp_g - ps.xLo_g)/(ps.nX*ps.nbX)),
      dy((ps.yUp_g - ps.yLo_g)/(ps.nY*ps.nbY)),
      dz((ps.zUp_g - ps.zLo_g)/(ps.nZ*ps.nbZ)),
      xLo(ps.xLo_g + dx*ps.layoutMPI.bxLo),
      xUp(ps.xLo_g + dx*ps.layoutMPI.bxUp),
      yLo(ps.yLo_g + dy*ps.layoutMPI.byLo),
      yUp(ps.yLo_g + dy*ps.layoutMPI.byUp),
      zLo(ps.zLo_g + dz*ps.layoutMPI.bzLo),
      zUp(ps.zLo_g + dz*ps.layoutMPI.bzUp),
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
      return dx*(Real(n) + Real(0.5)) + xLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real yCoord(idx_t n) const
    {
      return dy*(Real(n) + Real(0.5)) + yLo;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real zCoord(idx_t n) const
    {
      return dz*(Real(n) + Real(0.5)) + zLo;
    }

  private:
    Geometry() = delete;
  };
}

#endif
