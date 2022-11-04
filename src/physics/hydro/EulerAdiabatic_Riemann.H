#ifndef EULERADIABATIC_RIEMANN_H_
#define EULERADIABATIC_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "EulerAdiabatic_SimVar.H"
#include "EulerAdiabatic_EOS.H"
#include "../EquationTypes.H"
#include "../Riemann.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    void eulerFlux(Real gamma,int ord[3],const Real u[NUM_VARS],Real F[NUM_VARS])
    {
      Real p = EOS<EquationType::Euler>::pres(gamma,u);
      F[EulerAdiabatic::SimVar::dens] = u[ord[0]];                                       // rho u
      F[ord[0]      ] = u[ord[0]]*u[ord[0]]/u[EulerAdiabatic::SimVar::dens] + p;         // rho uu + p
      F[ord[1]      ] = u[ord[0]]*u[ord[1]]/u[EulerAdiabatic::SimVar::dens];             // rho uv
      F[ord[2]      ] = u[ord[0]]*u[ord[2]]/u[EulerAdiabatic::SimVar::dens];             // rho uw
      F[EulerAdiabatic::SimVar::etot] = u[ord[0]]*(u[EulerAdiabatic::SimVar::etot] + p)/u[EulerAdiabatic::SimVar::dens]; // u(E + p)
    }
      
    // Wavespeed estimates
    KOKKOS_INLINE_FUNCTION
    void eulerSpeed(Real gamma,int lng,
                  const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                  Real pL,Real pR,
                  Real* sL,Real* sR)
    {
      using Kokkos::Experimental::sqrt;
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fmin;
      Real a = fmax(sqrt(gamma*pL/uL[EulerAdiabatic::SimVar::dens]),sqrt(gamma*pR/uR[EulerAdiabatic::SimVar::dens]));
      *sL = fmin(uL[lng]/uL[EulerAdiabatic::SimVar::dens],uR[lng]/uR[EulerAdiabatic::SimVar::dens]) - a;
      *sR = fmax(uL[lng]/uL[EulerAdiabatic::SimVar::dens],uR[lng]/uR[EulerAdiabatic::SimVar::dens]) + a;
    }
      
    KOKKOS_INLINE_FUNCTION
    Real eulerHLL(Real gamma,int ord[3],
                  const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                  Real F[NUM_VARS])
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      Real FT[NUM_VARS];
      Real pL = EOS<EquationType::Euler>::pres(gamma,uL);
      Real pR = EOS<EquationType::Euler>::pres(gamma,uR);
      Real sL,sR;
      /* Wave speed estimates */
      eulerSpeed(gamma,ord[0],uL,uR,pL,pR,&sL,&sR);
      /* Find appropriate region and fill in flux vector */
      if(sR > 0) {
        eulerFlux(gamma,ord,uL,F);
        if(sL <= 0) {
          eulerFlux(gamma,ord,uR,FT);
          F[EulerAdiabatic::SimVar::dens] = hll(sL,sR,
                                F[EulerAdiabatic::SimVar::dens ],FT[EulerAdiabatic::SimVar::dens],
                                uL[EulerAdiabatic::SimVar::dens],uR[EulerAdiabatic::SimVar::dens]);
          F[ord[0]] =   hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
          F[ord[1]] =   hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
          F[ord[2]] =   hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
          F[EulerAdiabatic::SimVar::etot] = hll(sL,sR,
                                F[EulerAdiabatic::SimVar::etot ],FT[EulerAdiabatic::SimVar::etot],
                                uL[EulerAdiabatic::SimVar::etot],uR[EulerAdiabatic::SimVar::etot]);
        }
      } else {
        eulerFlux(gamma,ord,uR,F);
      }

      return fmax(fabs(sL),fabs(sR));
    }

    KOKKOS_INLINE_FUNCTION
    Real EulerHLL_X(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {EulerAdiabatic::SimVar::momx,EulerAdiabatic::SimVar::momy,EulerAdiabatic::SimVar::momz};
      return eulerHLL(gamma,ord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real EulerHLL_Y(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {EulerAdiabatic::SimVar::momy,EulerAdiabatic::SimVar::momz,EulerAdiabatic::SimVar::momx};
      return eulerHLL(gamma,ord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real EulerHLL_Z(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {EulerAdiabatic::SimVar::momz,EulerAdiabatic::SimVar::momx,EulerAdiabatic::SimVar::momy};
      return eulerHLL(gamma,ord,uL,uR,F);
    }
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
