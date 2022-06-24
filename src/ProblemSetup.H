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
    static constexpr int rad = STENCIL_RADIUS;
    Real gp_lFac,gp_bpow;

    // Solution file settings
    std::string baseName;
    
    // Domain and grid size
    int nX,nY,nZ;
    Real xLo,xUp,yLo,yUp,zLo,zUp;
    Real dx,dy,dz;

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
                 int a_nX,int a_nY,int a_nZ,
                 Real a_xLo,Real a_xUp,Real a_yLo,Real a_yUp,Real a_zLo,Real a_zUp,
                 Real gam):
      gp_lFac(Real(12.0)),
      gp_bpow(Real(STENCIL_RADIUS)),
      baseName(a_name),
      nX(a_nX),
      nY(a_nY),
      nZ(SPACE_DIM==3 ? a_nZ : 1),
      xLo(a_xLo),
      xUp(a_xUp),
      yLo(a_yLo),
      yUp(a_yUp),
      zLo(a_zLo),
      zUp(a_zUp),
      dx((xUp - xLo)/nX),
      dy((yUp - yLo)/nY),
      dz((zUp - zLo)/nZ),
      maxTimeSteps(100),
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
