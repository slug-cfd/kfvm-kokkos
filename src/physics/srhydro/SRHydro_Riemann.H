#ifndef SRHYDRO_RIEMANN_H_
#define SRHYDRO_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "SRHydro_SimVar.H"
#include "SRHydro_EOS.H"
#include "../EquationTypes.H"
#include "../Riemann.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    void srhydroFlux(Real gamma,int ord[3],
                     const Real u[NUM_VARS],const Real v[NUM_AUX],
                     Real F[NUM_VARS])
    {
      using SRHydro::SimVar;
      using SRHydro::AuxVar;
      
      F[SimVar::dens] = u[SimVar::dens]*v[ord[0]];
      F[ord[0]] = u[ord[0]]*v[ord[0]] + v[AuxVar::pres];
      F[ord[1]] = u[ord[1]]*v[ord[0]];
      F[ord[2]] = u[ord[2]]*v[ord[0]];
      F[SimVar::tau] = u[ord[0]] - u[SimVar::dens]*v[ord[0]];
    }
      
    // Wavespeed estimates
    KOKKOS_INLINE_FUNCTION
    void srhydroSpeed(Real gamma,int lng,
                      const Real vL[NUM_AUX],const Real vR[NUM_AUX],
                      Real* sL,Real* sR)
    {
      using Kokkos::fmax;
      using Kokkos::fmin;
      using Kokkos::sqrt;
      using SRHydro::AuxVar;

      // Get soundspeeds
      const Real aL = EOS<EquationType::SRHydro>::soundspeed(gamma,vL);
      const Real aR = EOS<EquationType::SRHydro>::soundspeed(gamma,vR);

      // Left state intermediate quantities
      const Real vsqL =
	vL[AuxVar::velx]*vL[AuxVar::velx] +
	vL[AuxVar::vely]*vL[AuxVar::vely] +
	vL[AuxVar::velz]*vL[AuxVar::velz];
      const Real asqL = Real(1.0) - aL*aL;
      const Real avsqL = Real(1.0) - aL*aL*vsqL;
      const Real sqrtL= aL*sqrt((Real(1.0) - vsqL)*(avsqL - vL[lng]*vL[lng]*asqL));

      // Right state intermediate quantities
      const Real vsqR =
	vR[AuxVar::velx]*vR[AuxVar::velx] +
	vR[AuxVar::vely]*vR[AuxVar::vely] +
	vR[AuxVar::velz]*vR[AuxVar::velz];
      const Real asqR = Real(1.0) - aR*aR;
      const Real avsqR = Real(1.0) - aR*aR*vsqR;
      const Real sqrtR= aR*sqrt((Real(1.0) - vsqR)*(avsqR - vR[lng]*vR[lng]*asqR));

      // Left going wavespeeds
      const Real sLm = (vL[lng]*asqL - sqrtL)/avsqL;
      const Real sRm = (vR[lng]*asqR - sqrtR)/avsqR;

      // Right going wavespeeds
      const Real sLp = (vL[lng]*asqL + sqrtL)/avsqL;
      const Real sRp = (vR[lng]*asqR + sqrtR)/avsqR;

      // Put it all together
      *sL = fmin(Real(0.0),fmin(sLm,sRm));
      *sR = fmax(Real(0.0),fmin(sLp,sRp));
    }
      
    KOKKOS_INLINE_FUNCTION
    Real srhydroHLL(Real gamma,int ord[3],
                    const Real uL[NUM_VARS],
                    const Real uR[NUM_VARS],
                    Real F[NUM_VARS])
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using SRHydro::SimVar;
      
      Real FT[NUM_VARS];
      Real vL[NUM_AUX],vR[NUM_AUX];
      Real sL,sR;

      // Get primitive vectors for each state
      EOS<EquationType::SRHydro>::primvec(gamma,uL,vL);
      EOS<EquationType::SRHydro>::primvec(gamma,uR,vR);
      
      // Wave speed estimates
      srhydroSpeed(gamma,ord[0],uL,uR,&sL,&sR);
      
      // Find appropriate region and fill in flux vector
      if(sR > 0) {
        srhydroFlux(gamma,ord,uL,vL,F);
        if(sL <= 0) {
          srhydroFlux(gamma,ord,uR,vR,FT);
          F[SimVar::dens] = hll(sL,sR,F[SimVar::dens],FT[SimVar::dens],
                                uL[SimVar::dens],uR[SimVar::dens]);
          F[ord[0]] = hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
          F[ord[1]] = hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
          F[ord[2]] = hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
          F[SimVar::tau] = hll(sL,sR,F[SimVar::tau],FT[SimVar::tau],
                                uL[SimVar::tau],uR[SimVar::tau]);
        }
      } else {
        srhydroFlux(gamma,ord,uR,vR,F);
      }

      return fmax(fabs(sL),fabs(sR));
    }

    template<RSType rst> struct SRHydroRS {};

    template<>
    struct SRHydroRS<RSType::HLL>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return srhydroHLL(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return srhydroHLL(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return srhydroHLL(gamma,ord,uL,uR,F);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
