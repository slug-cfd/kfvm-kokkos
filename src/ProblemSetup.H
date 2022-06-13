#ifndef PROBLEMSETUP_H_
#define PROBLEMSETUP_H_

#include <array>
#include <cstdint>
#include <string>

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
    static constexpr uint16_t rad = STENCIL_RADIUS;
    Real gp_lFac,gp_bpow;

    // Solution file settings
    std::string baseName;
    
    // Domain and grid size
    uint16_t nX,nY,nZ,nX_loc,nY_loc,nZ_loc;
    Real xLo,xUp,yLo,yUp,zLo,zUp,dx,dy,dz;

    // Time stepping information
    std::size_t maxTimeSteps;
    Real initialTime,finalTime,initialDeltaT;
    Real cfl;

    // File output information
    std::size_t plotFreq;

    // Fluid properties array
    FluidProperties fluidProp;

    // Simple constructor
    // To be replaced by input file reader
    ProblemSetup(std::string a_name,
                 uint16_t a_nX,uint16_t a_nY,uint16_t a_nZ,
                 uint16_t a_nX_loc,uint16_t a_nY_loc,uint16_t a_nZ_loc,
                 Real a_xLo,Real a_xUp,Real a_yLo,Real a_yUp,Real a_zLo,Real a_zUp,
                 Real gam):
      gp_lFac(Real(12.0)),
      gp_bpow(Real(STENCIL_RADIUS)),
      baseName(a_name),
      nX(a_nX),
      nY(a_nY),
      nZ(a_nZ),
      nX_loc(a_nX_loc),
      nY_loc(a_nY_loc),
      nZ_loc(a_nZ_loc),
      xLo(a_xLo),
      xUp(a_xUp),
      yLo(a_yLo),
      yUp(a_yUp),
      zLo(a_zLo),
      zUp(a_zUp),
      dx((xUp - xLo)/nX),
      dy((yUp - yLo)/nY),
      dz((zUp - zLo)/nZ),
      maxTimeSteps(2000),
      initialTime(Real(0.0)),
      finalTime(Real(0.2)),
      initialDeltaT(Real(1.e-8)),
      cfl(Real(0.75)),
      plotFreq(20),
      fluidProp({gam})
      {
      }
  };
  
}
#endif
