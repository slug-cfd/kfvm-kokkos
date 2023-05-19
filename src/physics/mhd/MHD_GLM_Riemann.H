#ifndef MHD_GLM_RIEMANN_H_
#define MHD_GLM_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../SimVar.H"
#include "MHD_GLM_EOS.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    Real logMean(Real a,Real b)
    {
      using Kokkos::log;

      Real zeta = a/b,f = (zeta - Real(1.0))/(zeta + Real(1.0)),u = f*f;
      Real F = u < Real(0.01) ?
	Real(1.0) + u/Real(3.0) + u*u/Real(5.0) + u*u*u/Real(7.0) :
	Real(0.5)*log(zeta)/f;

      return Real(0.5)*(a + b)/F;
    }

    KOKKOS_INLINE_FUNCTION
    Real arithMean(Real a,Real b)
    {
      return Real(0.5)*(a + b);
    }

    KOKKOS_INLINE_FUNCTION
    void mhdGlmFlux(Real gamma,Real ch,int vord[3],int bord[3],
		    const MHD_GLM::ConsVec U,const MHD_GLM::PrimVec V,
                    MHD_GLM::ConsVec F)
    {
      using MHD_GLM::SimVar;
      using MHD_GLM::PrimVar;

      const Real p = V[PrimVar::pres];
      const Real uDb = (V[PrimVar::velx]*U[SimVar::magx] +
			V[PrimVar::vely]*U[SimVar::magy] +
			V[PrimVar::velz]*U[SimVar::magz]);
      
      F[SimVar::dens] = U[vord[0]];
      
      F[vord[0]] = U[vord[0]]*V[vord[0]] - U[bord[0]]*U[bord[0]] + p;
      F[vord[1]] = U[vord[0]]*V[vord[1]] - U[bord[0]]*U[bord[1]];
      F[vord[2]] = U[vord[0]]*V[vord[2]] - U[bord[0]]*U[bord[2]];
      
      F[SimVar::etot] = V[vord[0]]*(U[SimVar::etot] + p - Real(0.5)*U[SimVar::psi]*U[SimVar::psi])
        + U[bord[0]]*(ch*U[SimVar::psi] - uDb);
      
      F[bord[0]] = ch*U[SimVar::psi];
      F[bord[1]] = (V[vord[0]]*U[bord[1]] - V[vord[1]]*U[bord[0]]);
      F[bord[2]] = (V[vord[0]]*U[bord[2]] - V[vord[2]]*U[bord[0]]);
      
      F[SimVar::psi] = ch*U[bord[0]];
    }      

    KOKKOS_INLINE_FUNCTION
    void mhdGlmSpeed(Real gamma,int lng,int bord[3],
                     const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                     const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
		     const Real ch,Real& sL,Real& sR)
    {
      using Kokkos::fmax;
      using Kokkos::fmin;
      using MHD_GLM::SimVar;
      
      const Real cfL = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma,uL,uL[bord[0]]);
      const Real cfR = EOS<EquationType::MHD_GLM>::fastMagSonic(gamma,uR,uR[bord[0]]);

      const Real slMs = fmin(vL[lng] - cfL,vR[lng] - cfR);
      const Real slCh = fmin(vL[lng] - ch ,vR[lng] - ch);
      const Real srMs = fmax(vL[lng] + cfL,vR[lng] + cfR);
      const Real srCh = fmax(vL[lng] + ch ,vR[lng] + ch);
      
      sL = fmin(slMs,slCh);
      sR = fmax(srMs,srCh);
    }
    
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmKEPES(Real gamma,Real ch,int vord[3],int bord[3],
                     const MHD_GLM::ConsVec uL,const MHD_GLM::ConsVec uR,
                     const MHD_GLM::PrimVec vL,const MHD_GLM::PrimVec vR,
		     MHD_GLM::ConsVec F)
    {
      using Kokkos::fmax;
      using Kokkos::fabs;
      using Kokkos::sqrt;
      using MHD_GLM::SimVar;
      using MHD_GLM::PrimVar;
      
      Real sL,sR,sMax;
      
      // Wave speed estimates
      mhdGlmSpeed(gamma,vord[0],bord,uL,uR,vL,vR,ch,sL,sR);
      sMax = fmax(fabs(sL),fabs(sR));

      // Find far too many mean quantites
      Real rhoLn = logMean(uL[SimVar::dens],uR[SimVar::dens]);
      Real rhoAv = arithMean(uL[SimVar::dens],uR[SimVar::dens]);
      
      Real uAv = arithMean(vL[vord[0]],vR[vord[0]]);
      Real vAv = arithMean(vL[vord[1]],vR[vord[1]]);
      Real wAv = arithMean(vL[vord[2]],vR[vord[2]]);
      
      Real uuAv = arithMean(vL[vord[0]]*vL[vord[0]],vR[vord[0]]*vR[vord[0]]);
      Real vvAv = arithMean(vL[vord[1]]*vL[vord[1]],vR[vord[1]]*vR[vord[1]]);
      Real wwAv = arithMean(vL[vord[2]]*vL[vord[2]],vR[vord[2]]*vR[vord[2]]);
      
      Real bxAv = arithMean(uL[bord[0]],uR[bord[0]]);
      Real byAv = arithMean(uL[bord[1]],uR[bord[1]]);
      Real bzAv = arithMean(uL[bord[2]],uR[bord[2]]);
      
      Real bxxAv = arithMean(uL[bord[0]]*uL[bord[0]],uR[bord[0]]*uR[bord[0]]);
      Real byyAv = arithMean(uL[bord[1]]*uL[bord[1]],uR[bord[1]]*uR[bord[1]]);
      Real bzzAv = arithMean(uL[bord[2]]*uL[bord[2]],uR[bord[2]]*uR[bord[2]]);
      
      Real uBxAv = arithMean(vL[vord[0]]*uL[bord[0]],vR[vord[0]]*uR[bord[0]]);
      Real vByAv = arithMean(vL[vord[1]]*uL[bord[1]],vR[vord[1]]*uR[bord[1]]);
      Real wBzAv = arithMean(vL[vord[2]]*uL[bord[2]],vR[vord[2]]*uR[bord[2]]);
      
      Real uBxxAv = arithMean(vL[vord[0]]*uL[bord[0]]*uL[bord[0]],vR[vord[0]]*uR[bord[0]]*uR[bord[0]]);
      Real uByyAv = arithMean(vL[vord[0]]*uL[bord[1]]*uL[bord[1]],vR[vord[0]]*uR[bord[1]]*uR[bord[1]]);
      Real uBzzAv = arithMean(vL[vord[0]]*uL[bord[2]]*uL[bord[2]],vR[vord[0]]*uR[bord[2]]*uR[bord[2]]);
      
      Real psiAv = arithMean(uL[SimVar::psi],uR[SimVar::psi]);
      Real bxPsiAv = arithMean(uL[bord[0]]*uL[SimVar::psi],uR[bord[0]]*uR[SimVar::psi]);

      Real betaLn = logMean(uL[SimVar::dens]/(Real(2.0)*EOS<EquationType::MHD_GLM>::prsg(gamma,uL)),
			    uR[SimVar::dens]/(Real(2.0)*EOS<EquationType::MHD_GLM>::prsg(gamma,uR)));
      Real betaAv = arithMean(uL[SimVar::dens]/(Real(2.0)*EOS<EquationType::MHD_GLM>::prsg(gamma,uL)),
			      uR[SimVar::dens]/(Real(2.0)*EOS<EquationType::MHD_GLM>::prsg(gamma,uR)));
      Real pTil = Real(0.5)*rhoAv/betaAv;
      Real pHatTot = pTil + Real(0.5)*(bxxAv + byyAv + bzzAv);

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
      for (int nV=0; nV<NUM_VARS; nV++) {
	F[nV] -= Real(0.5)*sMax*(uR[nV] - uL[nV]);
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
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
