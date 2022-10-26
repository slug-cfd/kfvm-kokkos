#ifndef RIEMANN_H_
#define RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../SimVar.H"
#include "../Types.H"
#include "EOS.H"

namespace KFVM {

  namespace Hydro {

    namespace EulerAdiabatic {

      KOKKOS_INLINE_FUNCTION
      void eulerFlux(Real gamma,int ord[3],const Real u[NUM_VARS],Real F[NUM_VARS])
      {
        Real p = pres(gamma,u);
        F[SimVar::dens] = u[ord[0]];                                       // rho u
        F[ord[0]      ] = u[ord[0]]*u[ord[0]]/u[SimVar::dens] + p;         // rho uu + p
        F[ord[1]      ] = u[ord[0]]*u[ord[1]]/u[SimVar::dens];             // rho uv
        F[ord[2]      ] = u[ord[0]]*u[ord[2]]/u[SimVar::dens];             // rho uw
        F[SimVar::etot] = u[ord[0]]*(u[SimVar::etot] + p)/u[SimVar::dens]; // u(E + p)
      }
      
      // Wavespeed estimates
      KOKKOS_INLINE_FUNCTION
      Real roeAvg(Real rhoL,Real rhoR,Real qL,Real qR)
      {
	using Kokkos::Experimental::sqrt;
        return (qL*sqrt(rhoL) + qR*sqrt(rhoR))/(sqrt(rhoL) + sqrt(rhoR));
      }

      KOKKOS_INLINE_FUNCTION
      void maxSpeed(Real gamma,int lng,
		    const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		    Real pL,Real pR,
		    Real* sL,Real* sR)
      {
	using Kokkos::Experimental::sqrt;
	using Kokkos::Experimental::fmax;
	using Kokkos::Experimental::fmin;
        Real a = fmax(sqrt(gamma*pL/uL[SimVar::dens]),sqrt(gamma*pR/uR[SimVar::dens]));
        *sL = fmin(uL[lng]/uL[SimVar::dens],uR[lng]/uR[SimVar::dens]) - a;
        *sR = fmax(uL[lng]/uL[SimVar::dens],uR[lng]/uR[SimVar::dens]) + a;
      }

      KOKKOS_INLINE_FUNCTION
      Real hll(Real sL,Real sR,Real fL,Real fR,Real uL,Real uR)
      {
        return ((sR*fL - sL*fR) + sL*sR*(uR - uL))/(sR - sL);
      }

      KOKKOS_INLINE_FUNCTION
      Real hllState(Real sL,Real sR,Real fL,Real fR,Real uL,Real uR)
      {
        return ((sR*uR - sL*uL) + (fL - fR))/(sR - sL);
      }
      
      KOKKOS_INLINE_FUNCTION
      Real eulerHLL(Real gamma,int ord[3],
		    const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		    Real F[NUM_VARS])
      {
	using Kokkos::Experimental::fmax;
	using Kokkos::Experimental::fabs;
        Real FT[NUM_VARS];
        Real pL = pres(gamma,uL);
        Real pR = pres(gamma,uR);
        Real sL,sR;
        /* Wave speed estimates */
        maxSpeed(gamma,ord[0],uL,uR,pL,pR,&sL,&sR);
        /* Find appropriate region and fill in flux vector */
        if(sR > 0) {
          eulerFlux(gamma,ord,uL,F);
          if(sL <= 0) {
            eulerFlux(gamma,ord,uR,FT);
            F[SimVar::dens] = hll(sL,sR,
                                  F[SimVar::dens ],FT[SimVar::dens],
                                  uL[SimVar::dens],uR[SimVar::dens]);
            F[ord[0]] =   hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
            F[ord[1]] =   hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
            F[ord[2]] =   hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
            F[SimVar::etot] = hll(sL,sR,
                                  F[SimVar::etot ],FT[SimVar::etot],
                                  uL[SimVar::etot],uR[SimVar::etot]);
          }
        } else {
          eulerFlux(gamma,ord,uR,F);
        }

        return fmax(fabs(sL),fabs(sR));
      }

      KOKKOS_INLINE_FUNCTION
      Real HLL_X(const Real gamma,
		 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		 Real F[NUM_VARS])
      {
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return eulerHLL(gamma,ord,uL,uR,F);
      }

      KOKKOS_INLINE_FUNCTION
      Real HLL_Y(const Real gamma,
		 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		 Real F[NUM_VARS])
      {
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return eulerHLL(gamma,ord,uL,uR,F);
      }

      KOKKOS_INLINE_FUNCTION
      Real HLL_Z(const Real gamma,
		 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		 Real F[NUM_VARS])
      {
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return eulerHLL(gamma,ord,uL,uR,F);
      }
      
    } // end namespace EulerAdiabatic
    
  } // end namespace Hydro
  
} // end namespace KFVM

#endif
