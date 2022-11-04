#ifndef MHD_GLM_RIEMANN_H_
#define MHD_GLM_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "MHD_GLM_SimVar.H"
#include "MHD_GLM_EOS.H"
#include "../EquationTypes.H"
#include "../Riemann.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    void mhdGlmFlux(Real gamma,int ord[3],const Real u[NUM_VARS],Real F[NUM_VARS])
    {
      Real p = EOS<EquationType::MHD_GLM>::pres(gamma,u);
      F[MHD_GLM::SimVar::dens] = u[ord[0]];                                       // rho u
      F[ord[0]      ] = u[ord[0]]*u[ord[0]]/u[MHD_GLM::SimVar::dens] + p;         // rho uu + p
      F[ord[1]      ] = u[ord[0]]*u[ord[1]]/u[MHD_GLM::SimVar::dens];             // rho uv
      F[ord[2]      ] = u[ord[0]]*u[ord[2]]/u[MHD_GLM::SimVar::dens];             // rho uw
      F[MHD_GLM::SimVar::etot] = u[ord[0]]*(u[MHD_GLM::SimVar::etot] + p)/u[MHD_GLM::SimVar::dens]; // u(E + p)
    }      

    KOKKOS_INLINE_FUNCTION
    void mhdGlmSpeed(Real gamma,int lng,
                  const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                  Real pL,Real pR,
                  Real* sL,Real* sR)
    {
      using Kokkos::Experimental::sqrt;
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fmin;
      Real a = fmax(sqrt(gamma*pL/uL[MHD_GLM::SimVar::dens]),sqrt(gamma*pR/uR[MHD_GLM::SimVar::dens]));
      *sL = fmin(uL[lng]/uL[MHD_GLM::SimVar::dens],uR[lng]/uR[MHD_GLM::SimVar::dens]) - a;
      *sR = fmax(uL[lng]/uL[MHD_GLM::SimVar::dens],uR[lng]/uR[MHD_GLM::SimVar::dens]) + a;
    }
      
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmHLL(Real gamma,int ord[3],
                  const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                  Real F[NUM_VARS])
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      Real FT[NUM_VARS];
      Real pL = EOS<EquationType::MHD_GLM>::pres(gamma,uL);
      Real pR = EOS<EquationType::MHD_GLM>::pres(gamma,uR);
      Real sL,sR;
      /* Wave speed estimates */
      mhdGlmSpeed(gamma,ord[0],uL,uR,pL,pR,&sL,&sR);
      /* Find appropriate region and fill in flux vector */
      if(sR > 0) {
        mhdGlmFlux(gamma,ord,uL,F);
        if(sL <= 0) {
          mhdGlmFlux(gamma,ord,uR,FT);
          F[MHD_GLM::SimVar::dens] = hll(sL,sR,
                                F[MHD_GLM::SimVar::dens ],FT[MHD_GLM::SimVar::dens],
                                uL[MHD_GLM::SimVar::dens],uR[MHD_GLM::SimVar::dens]);
          F[ord[0]] =   hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
          F[ord[1]] =   hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
          F[ord[2]] =   hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
          F[MHD_GLM::SimVar::etot] = hll(sL,sR,
                                F[MHD_GLM::SimVar::etot ],FT[MHD_GLM::SimVar::etot],
                                uL[MHD_GLM::SimVar::etot],uR[MHD_GLM::SimVar::etot]);
        }
      } else {
        mhdGlmFlux(gamma,ord,uR,F);
      }

      return fmax(fabs(sL),fabs(sR));
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_X(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz};
      return mhdGlmHLL(gamma,ord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_Y(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx};
      return mhdGlmHLL(gamma,ord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_Z(const Real gamma,
               const Real uL[NUM_VARS],const Real uR[NUM_VARS],
               Real F[NUM_VARS])
    {
      int ord[] = {MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy};
      return mhdGlmHLL(gamma,ord,uL,uR,F);
    }
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
