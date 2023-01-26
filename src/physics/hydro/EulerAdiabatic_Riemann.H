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
      using EulerAdiabatic::SimVar;
      
      Real p = EOS<EquationType::Euler>::pres(gamma,u);
      F[SimVar::dens] = u[ord[0]];
      F[ord[0]] = u[ord[0]]*u[ord[0]]/u[SimVar::dens] + p;
      F[ord[1]] = u[ord[0]]*u[ord[1]]/u[SimVar::dens];
      F[ord[2]] = u[ord[0]]*u[ord[2]]/u[SimVar::dens];
      F[SimVar::etot] = u[ord[0]]*(u[SimVar::etot] + p)/u[SimVar::dens];
    }
      
    // Wavespeed estimates
    KOKKOS_INLINE_FUNCTION
    void eulerSpeed(Real gamma,int lng,
        	    const Real uL[NUM_VARS],const Real uR[NUM_VARS],
        	    Real* sL,Real* sR)
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fmin;
      using EulerAdiabatic::SimVar;
      
      Real a = fmax(EOS<EquationType::Euler>::soundspeed(gamma,uL),
        	    EOS<EquationType::Euler>::soundspeed(gamma,uR));
      *sL = fmin(uL[lng]/uL[SimVar::dens],uR[lng]/uR[SimVar::dens]) - a;
      *sR = fmax(uL[lng]/uL[SimVar::dens],uR[lng]/uR[SimVar::dens]) + a;
    }
      
    KOKKOS_INLINE_FUNCTION
    Real eulerHLL(Real gamma,int ord[3],
                  const Real uL[NUM_VARS],
                  const Real uR[NUM_VARS],
                  Real F[NUM_VARS])
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      using EulerAdiabatic::SimVar;
      
      Real FT[NUM_VARS];
      Real sL,sR;
      
      // Wave speed estimates
      eulerSpeed(gamma,ord[0],uL,uR,&sL,&sR);
      
      // Find appropriate region and fill in flux vector
      if(sR > 0) {
        eulerFlux(gamma,ord,uL,F);
        if(sL <= 0) {
          eulerFlux(gamma,ord,uR,FT);
          F[SimVar::dens] = hll(sL,sR,F[SimVar::dens],FT[SimVar::dens],
                                uL[SimVar::dens],uR[SimVar::dens]);
          F[ord[0]] = hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
          F[ord[1]] = hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
          F[ord[2]] = hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
          F[SimVar::etot] = hll(sL,sR,F[SimVar::etot],FT[SimVar::etot],
                                uL[SimVar::etot],uR[SimVar::etot]);
        }
      } else {
        eulerFlux(gamma,ord,uR,F);
      }

      return fmax(fabs(sL),fabs(sR));
    }
      
    KOKKOS_INLINE_FUNCTION
    Real eulerHLLC(Real gamma,int ord[3],
		   const Real uL[NUM_VARS],
		   const Real uR[NUM_VARS],
		   Real F[NUM_VARS])
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      using EulerAdiabatic::SimVar;
      
      // Wave speed estimates
      Real sL,sR,sS;
      Real pL = EOS<EquationType::Euler>::pres(gamma,uL);
      Real pR = EOS<EquationType::Euler>::pres(gamma,uR);
      eulerSpeed(gamma,ord[0],uL,uR,&sL,&sR);
      sS = hllcSStar(uL[SimVar::dens],uR[SimVar::dens],pL,pR,
                     uL[ord[0]]/uL[SimVar::dens],uR[ord[0]]/uR[SimVar::dens],sL,sR);
      
      // Find correct zone
      if(0 < sL) {
        eulerFlux(gamma,ord,uL,F);
      } else if(0 <= sS) {
        eulerFlux(gamma,ord,uL,F);
        Real pre = hllcPre(uL[SimVar::dens],uL[ord[0]]/uL[SimVar::dens],sL,sS);
        F[SimVar::dens] += sL*(pre - uL[SimVar::dens]);
        F[ord[0]] += sL*(pre*sS - uL[ord[0]]);
        F[ord[1]] += sL*(pre*uL[ord[1]]/uL[SimVar::dens] - uL[ord[1]]);
        F[ord[2]] += sL*(pre*uL[ord[2]]/uL[SimVar::dens] - uL[ord[2]]);
        Real eStar = hllcEnergy(pre,uL[SimVar::dens],uL[SimVar::etot],pL,uL[ord[0]]/uL[SimVar::dens],sL,sS);
        F[SimVar::etot] += sL*(eStar - uL[SimVar::etot]);
      } else if(0 <= sR) {
        eulerFlux(gamma,ord,uR,F);
        Real pre = hllcPre(uR[SimVar::dens],uR[ord[0]]/uR[SimVar::dens],sR,sS);
        F[SimVar::dens] += sR*(pre - uR[SimVar::dens]);
        F[ord[0]] += sR*(pre*sS - uR[ord[0]]);
        F[ord[1]] += sR*(pre*uR[ord[1]]/uR[SimVar::dens] - uR[ord[1]]);
        F[ord[2]] += sR*(pre*uR[ord[2]]/uR[SimVar::dens] - uR[ord[2]]);
        Real eStar = hllcEnergy(pre,uR[SimVar::dens],uR[SimVar::etot],pR,uR[ord[0]]/uR[SimVar::dens],sR,sS);
        F[SimVar::etot] += sR*(eStar - uR[SimVar::etot]);
      } else {
        eulerFlux(gamma,ord,uR,F);
      }
  
      return fmax(fabs(sL),fabs(sR));
    }

    // Carbuncle suppressed Roe solver
    // As given by Hu and Feng @ https://doi.org/10.1016/j.cnsns.2022.106798
    KOKKOS_INLINE_FUNCTION
    Real eulerROE(Real gamma,int ord[3],
                  const Real uL[NUM_VARS],
                  const Real uR[NUM_VARS],
                  Real F[NUM_VARS])
    {
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fmin;
      using Kokkos::Experimental::fabs;
      using Kokkos::Experimental::exp;
      using EulerAdiabatic::SimVar;
      
      Real FT[NUM_VARS];

      // Soundspeed and Roe averaged values
      Real aL = EOS<EquationType::Euler>::soundspeed(gamma,uL);
      Real aR = EOS<EquationType::Euler>::soundspeed(gamma,uR);
      Real pL = EOS<EquationType::Euler>::pres(gamma,uL);
      Real pR = EOS<EquationType::Euler>::pres(gamma,uR);
      Real rhoL = uL[SimVar::dens],rhoR = uR[SimVar::dens],rTil = roeAvg(rhoL,rhoR,rhoL,rhoR);
      Real aTil = roeAvg(rhoL,rhoR,aL,aR);
      Real pTil = roeAvg(rhoL,rhoR,pL,pR);
      Real uLngL = uL[ord[0]]/rhoL,uLngR = uR[ord[0]]/rhoR;
      Real uTil = roeAvg(rhoL,rhoR,uLngL,uLngR);
      Real vTil = roeAvg(rhoL,rhoR,uL[ord[1]]/rhoL,uR[ord[1]]/rhoR);
      Real wTil = roeAvg(rhoL,rhoR,uL[ord[2]]/rhoL,uR[ord[2]]/rhoR);
      Real machL = uLngL/aL,machR = uLngR/aR;
      Real mTil = fabs(roeAvg(rhoL,rhoR,machL,machR));

      // Jumps
      Real drho = rhoR - rhoL;
      Real dv = uR[ord[1]]/rhoR - uL[ord[1]]/rhoL,dw = uR[ord[2]]/rhoR - uL[ord[2]]/rhoL;
      Real dp = pR - pL;
      
      // Wave speed estimates
      Real sL = fmin(Real(0.0),uTil - aTil),sR = fmax(Real(0.0),uTil + aTil);

      // Left and right fluxes
      eulerFlux(gamma,ord,uL,F);
      eulerFlux(gamma,ord,uR,FT);

      // Centered HLL
      F[SimVar::dens] = hll(sL,sR,F[SimVar::dens],FT[SimVar::dens],uL[SimVar::dens],uR[SimVar::dens]);
      F[ord[0]] = hll(sL,sR,F[ord[0]],FT[ord[0]],uL[ord[0]],uR[ord[0]]);
      F[ord[1]] = hll(sL,sR,F[ord[1]],FT[ord[1]],uL[ord[1]],uR[ord[1]]);
      F[ord[2]] = hll(sL,sR,F[ord[2]],FT[ord[2]],uL[ord[2]],uR[ord[2]]);
      F[SimVar::etot] = hll(sL,sR,F[SimVar::etot],FT[SimVar::etot],uL[SimVar::etot],uR[SimVar::etot]);

      // Roe antidiffusion
      Real pdt = dp/pTil;
      Real pre = exp(-pdt*pdt)*sL*sR/((Real(1.0) + mTil)*(sR - sL));
      Real AEW = pre*(drho - dp/(aTil*aTil));
      Real ASWv = pre*rTil*dv,ASWw = pre*rTil*dw;
      F[SimVar::dens] -= AEW;
      F[ord[0]] -= AEW*uTil;
      F[ord[1]] -= (AEW*vTil + ASWv);
      F[ord[2]] -= (AEW*wTil + ASWw);
      F[SimVar::etot] -= (AEW*(uTil*uTil + vTil*vTil + wTil*wTil)/Real(2.0) + ASWv*vTil + ASWw*wTil);

      return fmax(fabs(sL),fabs(sR));
    }

    template<RSType rst> struct EulerRS {};

    template<>
    struct EulerRS<RSType::HLL>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return eulerHLL(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return eulerHLL(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return eulerHLL(gamma,ord,uL,uR,F);
      }
    };

    template<>
    struct EulerRS<RSType::HLLC>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return eulerHLLC(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return eulerHLLC(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return eulerHLLC(gamma,ord,uL,uR,F);
      }
    };

    template<>
    struct EulerRS<RSType::ROE>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return eulerROE(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return eulerROE(gamma,ord,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const Real uL[NUM_VARS],const Real uR[NUM_VARS],
                          Real F[NUM_VARS])
      {
        using EulerAdiabatic::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return eulerROE(gamma,ord,uL,uR,F);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
