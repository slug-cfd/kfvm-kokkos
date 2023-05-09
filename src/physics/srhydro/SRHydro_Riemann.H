#ifndef SRHYDRO_RIEMANN_H_
#define SRHYDRO_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../SimVar.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "SRHydro_EOS.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    void srhydroFlux(const Real gamma,int ord[3],
                     const SRHydro::ConsVec u,const SRHydro::PrimVec v,
                     SRHydro::ConsVec F)
    {
      using SRHydro::SimVar;
      using SRHydro::PrimVar;
      
      F[SimVar::dens] = u[SimVar::dens]*v[ord[0]];
      F[ord[0]] = u[ord[0]]*v[ord[0]] + v[PrimVar::pres];
      F[ord[1]] = u[ord[1]]*v[ord[0]];
      F[ord[2]] = u[ord[2]]*v[ord[0]];
      F[SimVar::tau] = u[ord[0]] - u[SimVar::dens]*v[ord[0]];
    }
      
    // Wavespeed estimates
    KOKKOS_INLINE_FUNCTION
    void srhydroSpeed(Real gamma,int lng,
                      const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                      Real& sL,Real& sR)
    {
      using Kokkos::fmax;
      using Kokkos::fmin;
      using Kokkos::sqrt;
      using SRHydro::PrimVar;

      // Get soundspeeds
      const Real aL = EOS<EquationType::SRHydro>::soundspeed(gamma,vL),aLsq = aL*aL;
      const Real aR = EOS<EquationType::SRHydro>::soundspeed(gamma,vR),aRsq = aR*aR;

      // Left state intermediate quantities
      const Real wL = vL[PrimVar::lorz],sigL = aLsq/(wL*wL*(Real(1.0) - aLsq));
      const Real sqrtL= sqrt(sigL*(Real(1.0) + sigL - vL[lng]*vL[lng]));
      
      // Right state intermediate quantities
      const Real wR = vR[PrimVar::lorz],sigR = aRsq/(wR*wR*(Real(1.0) - aRsq));
      const Real sqrtR= sqrt(sigR*(Real(1.0) + sigR - vR[lng]*vR[lng]));

      // Left going wavespeeds
      const Real sLm = (vL[lng] - sqrtL)/(Real(1.0) + sigL);
      const Real sRm = (vR[lng] - sqrtR)/(Real(1.0) + sigR);

      // Right going wavespeeds
      const Real sLp = (vL[lng] + sqrtL)/(Real(1.0) + sigL);
      const Real sRp = (vR[lng] + sqrtR)/(Real(1.0) + sigR);

      // Put it all together
      sL = fmin(sLm,sRm);
      sR = fmin(sLp,sRp);
    }
      
    KOKKOS_INLINE_FUNCTION
    Real srhydroHLL(Real gamma,int ord[3],
                    const SRHydro::ConsVec uL,
                    const SRHydro::ConsVec uR,
                    const SRHydro::PrimVec vL,
                    const SRHydro::PrimVec vR,
                    SRHydro::ConsVec F)
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using SRHydro::SimVar;
      using SRHydro::PrimVar;
      using SRHydro::ConsVec;
      
      ConsVec FT;
      Real sL,sR;
      
      // Wave speed estimates
      srhydroSpeed(gamma,ord[0],vL,vR,sL,sR);
      
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

    // Refer to Mignone and Bodo doi:10.1111/j.1365-2966.2005.09546.x
    // Implementation based on Pluto code
    KOKKOS_INLINE_FUNCTION
    Real srhydroHLLC(Real gamma,int ord[3],
                    const SRHydro::ConsVec uL,
                    const SRHydro::ConsVec uR,
                    const SRHydro::PrimVec vL,
                    const SRHydro::PrimVec vR,
                    SRHydro::ConsVec F)
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using Kokkos::sqrt;
      using SRHydro::SimVar;
      using SRHydro::PrimVar;
      using SRHydro::ConsVec;
      
      Real sL,sR;
      
      // Wave speed estimates
      srhydroSpeed(gamma,ord[0],vL,vR,sL,sR);

      // Bail early if possible
      if (sL >= 0) {
        srhydroFlux(gamma,ord,uL,vL,F);
        return fmax(fabs(sL),fabs(sR));
      } else if (sR <= 0) {
        srhydroFlux(gamma,ord,uR,vR,F);
        return fmax(fabs(sL),fabs(sR));
      }

      // Must be in the intermediate region
      // Evaluate flux for each state
      ConsVec FT;
      srhydroFlux(gamma,ord,uL,vL,F);
      srhydroFlux(gamma,ord,uR,vR,FT);

      // Solve for contact speed and pressure
      const Real eL = uL[SimVar::tau] + uL[SimVar::dens],feL = uL[ord[0]];
      const Real eR = uR[SimVar::tau] + uR[SimVar::dens],feR = uR[ord[0]];

      const Real aL = sL*eL - feL,aR = sR*eR - feR;
      const Real bL = sL*uL[ord[0]] - F[ord[0]] - vL[PrimVar::pres];
      const Real bR = sR*uR[ord[0]] - FT[ord[0]] - vR[PrimVar::pres];

      const Real a = aR*sL - aL*sR,b = aL - aR + bL*sR - bR*sL,c = bR - bL;
      const Real sb = b>0 ? Real(1.0) : -Real(1.0);
      const Real sS = -Real(2.0)*c/(b + sb*sqrt(b*b - Real(4.0)*a*c));
      const Real pS = (aL*sS - bL)/(Real(1.0) - sS*sL);

      // Test which side we are on and fill in flux
      if (sS > 0) {
	const Real pf = (sL - vL[ord[0]])/(sL - sS);
	const Real Ds = pf*uL[SimVar::dens];
	const Real Ms0 = sS*(sL*(eL + pS) - uL[ord[0]])/(sL - sS);
	const Real Es = eL + (Ms0 - uL[ord[0]])/sL;
	const Real Ms1 = pf*uL[ord[1]],Ms2 = pf*uL[ord[2]];
	F[SimVar::dens] = F[SimVar::dens] + sL*(Ds  - uL[SimVar::dens]);
	F[ord[0]]       = F[ord[0]]       + sL*(Ms0 - uL[ord[0]]      );
	F[ord[1]]       = F[ord[1]]       + sL*(Ms1 - uL[ord[1]]      );
	F[ord[2]]       = F[ord[2]]       + sL*(Ms2 - uL[ord[2]]      );
	const Real feS  = feL             + sL*(Es  - eL              );
	F[SimVar::tau] = feS - F[SimVar::dens];
      } else {
	const Real pf = (sR - vR[ord[0]])/(sR - sS);
	const Real Ds = pf*uR[SimVar::dens];
	const Real Ms0 = sS*(sR*(eR + pS) - uR[ord[0]])/(sR - sS);
	const Real Es = eR + (Ms0 - uR[ord[0]])/sR;
	const Real Ms1 = pf*uR[ord[1]],Ms2 = pf*uR[ord[2]];
	F[SimVar::dens] = F[SimVar::dens] + sR*(Ds  - uR[SimVar::dens]);
	F[ord[0]]       = F[ord[0]]       + sR*(Ms0 - uR[ord[0]]      );
	F[ord[1]]       = F[ord[1]]       + sR*(Ms1 - uR[ord[1]]      );
	F[ord[2]]       = F[ord[2]]       + sR*(Ms2 - uR[ord[2]]      );
	const Real feS  = feR             + sR*(Es  - eR              );
	F[SimVar::tau] = feS - F[SimVar::dens];
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
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return srhydroHLL(gamma,ord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return srhydroHLL(gamma,ord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return srhydroHLL(gamma,ord,uL,uR,vL,vR,F);
      }
    };

    template<>
    struct SRHydroRS<RSType::HLLC>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momx,SimVar::momy,SimVar::momz};
        return srhydroHLLC(gamma,ord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momy,SimVar::momz,SimVar::momx};
        return srhydroHLLC(gamma,ord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const SRHydro::ConsVec uL,const SRHydro::ConsVec uR,
                          const SRHydro::PrimVec vL,const SRHydro::PrimVec vR,
                          SRHydro::ConsVec F)
      {
        using SRHydro::SimVar;
        int ord[] = {SimVar::momz,SimVar::momx,SimVar::momy};
        return srhydroHLLC(gamma,ord,uL,uR,vL,vR,F);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
