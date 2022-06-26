#ifndef PROBLEMSETUP_H_
#define PROBLEMSETUP_H_

#include <array>
#include <cstdint>
#include <string>

#include "Dimension.H"
#include "Types.H"

#ifndef STENCIL_RADIUS
#define STENCIL_RADIUS 2
#warning Defaulted to radius 2 stencil
#endif

namespace KFVM {

  struct ProblemSetup
  {
    // Static members to be set by the user
    static std::array<std::string,NUM_VARS> varName;
    
    // Stencil settings
    static constexpr idx_t rad = STENCIL_RADIUS;
    Real gp_lFac,gp_bpow;

    // Solution file settings
    std::string baseName;
    
    // Domain and grid size
    idx_t nX,nY,nZ;
    Real xLo,xUp,yLo,yUp,zLo,zUp;
    Real dx,dy,dz;

    // Time stepping information
    idx_t maxTimeSteps;
    Real initialTime,finalTime,initialDeltaT;
    Real cfl;

    // File output information
    idx_t plotFreq;

    // Fluid properties array
    FluidProperties fluidProp;

    // Simple constructor
    // To be replaced by input file reader
    ProblemSetup(std::string name_,
                 idx_t nX_,idx_t nY_,idx_t nZ_,
                 Real xLo_,Real xUp_,Real yLo_,Real yUp_,Real zLo_,Real zUp_,
		 Real finalTime_,
                 Real gam_):
      gp_lFac(Real(12.0)),
      gp_bpow(Real(STENCIL_RADIUS)),
      baseName(name_),
      nX(nX_),
      nY(nY_),
      nZ(SPACE_DIM==3 ? nZ_ : 1),
      xLo(xLo_),
      xUp(xUp_),
      yLo(yLo_),
      yUp(yUp_),
      zLo(zLo_),
      zUp(zUp_),
      dx((xUp - xLo)/nX),
      dy((yUp - yLo)/nY),
      dz((zUp - zLo)/nZ),
      maxTimeSteps(10000),
      initialTime(Real(0.0)),
      finalTime(finalTime_),
      initialDeltaT(Real(1.e-8)),
      cfl(SPACE_DIM==3 ? 0.5 : 0.75),
      plotFreq(50),
      fluidProp({gam_})
      {
      }
  };
  
}
#endif
