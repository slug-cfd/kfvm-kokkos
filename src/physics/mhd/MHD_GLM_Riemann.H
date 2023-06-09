#ifndef MHD_GLM_RIEMANN_H_
#define MHD_GLM_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../SimVar.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "MHD_GLM_EOS.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    Real logMean(const Real a,const Real b)
    {
      using Kokkos::log;

      const Real zeta = a/b,f = (zeta - Real(1.0))/(zeta + Real(1.0)),u = f*f;
      const Real F = u < Real(0.01) ?
	Real(1.0) + u/Real(3.0) + u*u/Real(5.0) + u*u*u/Real(7.0) :
	Real(0.5)*log(zeta)/f;

      return Real(0.5)*(a + b)/F;
    }

    KOKKOS_INLINE_FUNCTION
    Real arithMean(const Real a,const Real b)
    {
      return Real(0.5)*(a + b);
    }

    KOKKOS_INLINE_FUNCTION
    Real mhdGlmSpeed(const Real gamma,int lng,int bord[3],
                     const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                     const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
		     const Real ch,Real& sL,Real& sR)
    {
      using Kokkos::fmax;
      using Kokkos::fmin;
      using Kokkos::fabs;
      using MHD_GLM::SimVar;
      
      const Real cfL = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma,uL,uL[bord[0]]);
      const Real cfR = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma,uR,uR[bord[0]]);

      const Real slMs = fmin(vL[lng] - cfL,vR[lng] - cfR);
      const Real slCh = fmin(vL[lng] - ch ,vR[lng] - ch);
      const Real srMs = fmax(vL[lng] + cfL,vR[lng] + cfR);
      const Real srCh = fmax(vL[lng] + ch ,vR[lng] + ch);
      
      sL = fmin(slMs,slCh);
      sR = fmax(srMs,srCh);
      
      return fmax(fabs(sL),fabs(sR));
    }

    KOKKOS_INLINE_FUNCTION
    void mhdGlmFlux(const Real gamma,const Real ch,int vord[3],int bord[3],
		    const MHD_GLM::ConsVec U,const MHD_GLM::PrimVec V,
		    MHD_GLM::ConsVec F)
    {
      using MHD_GLM::SimVar;
      using MHD_GLM::PrimVar;

      const Real mx = U[vord[0]];
      const Real vx = V[vord[0]],vy = V[vord[1]],vz = V[vord[2]];
      const Real bx = U[bord[0]],by = U[bord[1]],bz = U[bord[2]];
      const Real et = U[SimVar::etot],pg = V[PrimVar::prsg];
      const Real chpsi = ch*U[SimVar::psi],hpsisq = 0.5*U[SimVar::psi]*U[SimVar::psi];
      const Real uDb = vx*bx + vy*by + vz*bz,bsq = bx*bx + by*by + bz*bz;
      const Real pt = pg + 0.5*bsq;

      F[SimVar::dens] = mx;
      F[vord[0]] = mx*vx + pt - bx*bx;
      F[vord[1]] = mx*vy - bx*by;
      F[vord[2]] = mx*vz - bx*bz;
      F[SimVar::etot] = vx*(et + pt - hpsisq) + bx*(chpsi - uDb);
      F[bord[0]] = chpsi;
      F[bord[1]] = vx*by - vy*bx;
      F[bord[2]] = vx*bz - vz*bx;
      F[SimVar::psi] = ch*bx;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmLLF(const Real gamma,const Real ch,int vord[3],int bord[3],
		   const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
		   const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
		   MHD_GLM::ConsVec F)
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using MHD_GLM::ConsVec;
      
      // Wave speed estimates
      Real sL,sR;
      const Real sMax = mhdGlmSpeed(gamma,vord[0],bord,uL,uR,vL,vR,ch,sL,sR);

      // Left and right fluxes
      ConsVec FR;
      mhdGlmFlux(gamma,ch,vord,bord,uL,vL,F);
      mhdGlmFlux(gamma,ch,vord,bord,uR,vR,FR);

      // Add in local Lax-Friedrichs dissipation
      // Total energy treated through entropy variables
      for (int nV=0; nV<NUM_VARS; nV++) {
	F[nV] = 0.5*((F[nV] + FR[nV]) - sMax*(uR[nV] - uL[nV]));
      }

      return sMax;
    }
    
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmKEPES(const Real gamma,const Real ch,int vord[3],int bord[3],
                     const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                     const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
		     MHD_GLM::ConsVec F)
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using Kokkos::sqrt;
      using MHD_GLM::SimVar;
      using MHD_GLM::PrimVar;
      
      
      // Wave speed estimates
      Real sL,sR;
      const Real sMax = mhdGlmSpeed(gamma,vord[0],bord,uL,uR,vL,vR,ch,sL,sR);

      // Find far too many mean quantites
      const Real rhoLn = logMean(uL[SimVar::dens],uR[SimVar::dens]);
      const Real rhoAv = arithMean(uL[SimVar::dens],uR[SimVar::dens]);
      
      const Real uAv = arithMean(vL[vord[0]],vR[vord[0]]);
      const Real vAv = arithMean(vL[vord[1]],vR[vord[1]]);
      const Real wAv = arithMean(vL[vord[2]],vR[vord[2]]);
      
      const Real uuAv = arithMean(vL[vord[0]]*vL[vord[0]],vR[vord[0]]*vR[vord[0]]);
      const Real vvAv = arithMean(vL[vord[1]]*vL[vord[1]],vR[vord[1]]*vR[vord[1]]);
      const Real wwAv = arithMean(vL[vord[2]]*vL[vord[2]],vR[vord[2]]*vR[vord[2]]);
      
      const Real bxAv = arithMean(uL[bord[0]],uR[bord[0]]);
      const Real byAv = arithMean(uL[bord[1]],uR[bord[1]]);
      const Real bzAv = arithMean(uL[bord[2]],uR[bord[2]]);
      
      const Real bxxAv = arithMean(uL[bord[0]]*uL[bord[0]],uR[bord[0]]*uR[bord[0]]);
      const Real byyAv = arithMean(uL[bord[1]]*uL[bord[1]],uR[bord[1]]*uR[bord[1]]);
      const Real bzzAv = arithMean(uL[bord[2]]*uL[bord[2]],uR[bord[2]]*uR[bord[2]]);
      
      const Real uBxAv = arithMean(vL[vord[0]]*uL[bord[0]],vR[vord[0]]*uR[bord[0]]);
      const Real vByAv = arithMean(vL[vord[1]]*uL[bord[1]],vR[vord[1]]*uR[bord[1]]);
      const Real wBzAv = arithMean(vL[vord[2]]*uL[bord[2]],vR[vord[2]]*uR[bord[2]]);
      
      const Real uBxxAv = arithMean(vL[vord[0]]*uL[bord[0]]*uL[bord[0]],vR[vord[0]]*uR[bord[0]]*uR[bord[0]]);
      const Real uByyAv = arithMean(vL[vord[0]]*uL[bord[1]]*uL[bord[1]],vR[vord[0]]*uR[bord[1]]*uR[bord[1]]);
      const Real uBzzAv = arithMean(vL[vord[0]]*uL[bord[2]]*uL[bord[2]],vR[vord[0]]*uR[bord[2]]*uR[bord[2]]);
      
      const Real psiAv = arithMean(uL[SimVar::psi],uR[SimVar::psi]);
      const Real bxPsiAv = arithMean(uL[bord[0]]*uL[SimVar::psi],uR[bord[0]]*uR[SimVar::psi]);

      const Real betaL = uL[SimVar::dens]/(Real(2.0)*vL[PrimVar::prsg]);
      const Real betaR = uR[SimVar::dens]/(Real(2.0)*vR[PrimVar::prsg]);
      const Real betaLn = logMean(betaL,betaR);
      const Real betaAv = arithMean(betaL,betaR);

      const Real pLn = logMean(vL[PrimVar::prsg],vR[PrimVar::prsg]);
      const Real pTil = Real(0.5)*rhoAv/betaAv;
      const Real pHatTot = pTil + Real(0.5)*(bxxAv + byyAv + bzzAv);

      // Fill in flux
      F[SimVar::dens] = rhoLn*uAv;
      
      F[vord[0]] = rhoLn*uAv*uAv + pHatTot - bxAv*bxAv;
      F[vord[1]] = rhoLn*uAv*vAv - bxAv*byAv;
      F[vord[2]] = rhoLn*uAv*wAv - bxAv*bzAv;
      
      F[bord[0]] = ch*psiAv;
      F[bord[1]] = uAv*byAv - vAv*bxAv;
      F[bord[2]] = uAv*bzAv - wAv*bxAv;

      F[SimVar::psi] = ch*bxAv;

      F[SimVar::etot] =
	(Real(1.0)/(Real(2.0)*(gamma - Real(1.0))*betaLn) - Real(0.5)*(uuAv + vvAv + wwAv))*F[SimVar::dens] +
	uAv*F[vord[0]] + vAv*F[vord[1]] + wAv*F[vord[2]] +
	bxAv*F[bord[0]] + byAv*F[bord[1]] + bzAv*F[bord[2]] +
	psiAv*F[SimVar::psi] -
	Real(0.5)*(uBxxAv + uByyAv + uBzzAv) +
	bxAv*(uBxAv + vByAv + wBzAv) -
	ch*bxPsiAv;

      // Add in local Lax-Friedrichs dissipation
      // Total energy treated through entropy variables
      for (int nV=0; nV<NUM_VARS; nV++) {
	F[nV] -= sMax*(uR[nV] - uL[nV])/Real(2.0);
      }

      return sMax;
    }

    template<RSType rst> struct MHD_GLM_RS {};

    template<>
    struct MHD_GLM_RS<RSType::MHD_GLM_KEPES>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz};
        int bord[] = {MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz};
        return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx};
        int bord[] = {MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx};
        return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy};
        int bord[] = {MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy};
        return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }
    };

    template<>
    struct MHD_GLM_RS<RSType::LLF>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz};
        int bord[] = {MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz};
        return mhdGlmLLF(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx};
        int bord[] = {MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx};
        return mhdGlmLLF(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,const Real ch,
                          const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                          const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
                          MHD_GLM::ConsVec F)
      {
        int vord[] = {MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy};
        int bord[] = {MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy};
        return mhdGlmLLF(gamma,ch,vord,bord,uL,uR,vL,vR,F);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
