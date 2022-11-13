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
    Real logMean(Real a,Real b)
    {
      using Kokkos::Experimental::log;

      Real zeta = a/b,f = (zeta - 1.0)/(zeta + 1.0),u = f*f;
      Real F = u < 0.01 ?
	1.0 + u/3.0 + u*u/5.0 + u*u*u/7.0 :
	log(zeta)/(2.0*f);

      return (a + b)/(2.0*F);
    }

    KOKKOS_INLINE_FUNCTION
    Real arithMean(Real a,Real b)
    {
      return (a + b)/2.0;
    }

    KOKKOS_INLINE_FUNCTION
    void mhdGlmFlux(Real gamma,Real ch,int vord[3],int bord[3],
		    const Real u[NUM_VARS],Real F[NUM_VARS])
    {
      using MHD_GLM::SimVar;

      const Real p = EOS<EquationType::MHD_GLM>::pres(gamma,u);
      const Real uDb = (u[SimVar::momx]*u[SimVar::magx] +
			u[SimVar::momy]*u[SimVar::magy] +
			u[SimVar::momz]*u[SimVar::magz])/u[SimVar::dens];
      
      F[SimVar::dens] = u[vord[0]];
      
      F[vord[0]] = u[vord[0]]*u[vord[0]]/u[SimVar::dens] - u[bord[0]]*u[bord[0]] + p;
      F[vord[1]] = u[vord[0]]*u[vord[1]]/u[SimVar::dens] - u[bord[0]]*u[bord[1]];
      F[vord[2]] = u[vord[0]]*u[vord[2]]/u[SimVar::dens] - u[bord[0]]*u[bord[2]];
      
      F[SimVar::etot] = u[vord[0]]*(u[SimVar::etot] + p)/u[SimVar::dens] + u[bord[0]]*(ch*u[SimVar::psi] - uDb);
      
      F[bord[0]] = ch*u[SimVar::psi];
      F[bord[1]] = (u[vord[0]]*u[bord[1]] - u[vord[1]]*u[bord[0]])/u[SimVar::dens];
      F[bord[2]] = (u[vord[0]]*u[bord[2]] - u[vord[2]]*u[bord[0]])/u[SimVar::dens];
      
      F[SimVar::psi] = ch*u[bord[0]];
    }      

    KOKKOS_INLINE_FUNCTION
    void mhdGlmSpeed(Real gamma,int lng,int bord[3],
                     const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		     const Real ch,Real* sL,Real* sR)
    {
      using MHD_GLM::SimVar;
      using Kokkos::Experimental::sqrt;
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fmin;

      const Real pgL = EOS<EquationType::MHD_GLM>::prsg(gamma,uL);
      const Real aCaL = (gamma*pgL + (uL[bord[0]]*uL[bord[0]] +
				      uL[bord[1]]*uL[bord[1]] +
				      uL[bord[2]]*uL[bord[2]]))/uL[SimVar::dens];
      const Real aCaxL = gamma*pgL*uL[bord[0]]*uL[bord[0]]/(uL[SimVar::dens]*uL[SimVar::dens]);
      const Real cfL = sqrt((aCaL + sqrt(aCaL*aCaL - 4.0*aCaxL))/2.0);

      const Real pgR = EOS<EquationType::MHD_GLM>::prsg(gamma,uR);
      const Real aCaR = (gamma*pgR + (uR[bord[0]]*uR[bord[0]] +
				      uR[bord[1]]*uR[bord[1]] +
				      uR[bord[2]]*uR[bord[2]]))/uR[SimVar::dens];
      const Real aCaxR = gamma*pgR*uR[bord[0]]*uR[bord[0]]/(uR[SimVar::dens]*uR[SimVar::dens]);
      const Real cfR = sqrt((aCaR + sqrt(aCaR*aCaR - 4.0*aCaxR))/2.0);

      const Real slMs = fmin(uL[lng]/uL[SimVar::dens] - cfL,uR[lng]/uR[SimVar::dens] - cfR);
      const Real slCh = fmin(uL[lng]/uL[SimVar::dens] - ch,uR[lng]/uR[SimVar::dens] - ch);
      const Real srMs = fmax(uL[lng]/uL[SimVar::dens] + cfL,uR[lng]/uR[SimVar::dens] + cfR);
      const Real srCh = fmax(uL[lng]/uL[SimVar::dens] + ch,uR[lng]/uR[SimVar::dens] + ch);
      
      *sL = fmin(slMs,slCh);
      *sR = fmax(srMs,srCh);
    }
      
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmHLL(Real gamma,Real ch,int vord[3],int bord[3],
		   const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		   Real F[NUM_VARS])
    {
      using MHD_GLM::SimVar;
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      
      Real FT[NUM_VARS];
      Real sL,sR;
      
      // Wave speed estimates
      mhdGlmSpeed(gamma,vord[0],bord,uL,uR,ch,&sL,&sR);
      
      // Find appropriate region and fill in flux vector
      if(sR > 0) {
        mhdGlmFlux(gamma,ch,vord,bord,uL,F);
        if(sL <= 0) {
          mhdGlmFlux(gamma,ch,vord,bord,uR,FT);
          F[SimVar::dens] = hll(sL,sR,F[SimVar::dens],FT[SimVar::dens],uL[SimVar::dens],uR[SimVar::dens]);
          F[vord[0]] = hll(sL,sR,F[vord[0]],FT[vord[0]],uL[vord[0]],uR[vord[0]]);
          F[vord[1]] = hll(sL,sR,F[vord[1]],FT[vord[1]],uL[vord[1]],uR[vord[1]]);
          F[vord[2]] = hll(sL,sR,F[vord[2]],FT[vord[2]],uL[vord[2]],uR[vord[2]]);
          F[SimVar::etot] = hll(sL,sR,F[SimVar::etot],FT[SimVar::etot],uL[SimVar::etot],uR[SimVar::etot]);
          F[bord[0]] = hll(sL,sR,F[bord[0]],FT[bord[0]],uL[bord[0]],uR[bord[0]]);
          F[bord[1]] = hll(sL,sR,F[bord[1]],FT[bord[1]],uL[bord[1]],uR[bord[1]]);
          F[bord[2]] = hll(sL,sR,F[bord[2]],FT[bord[2]],uL[bord[2]],uR[bord[2]]);
          F[SimVar::psi] = hll(sL,sR,F[SimVar::psi],FT[SimVar::psi],uL[SimVar::psi],uR[SimVar::psi]);
        }
      } else {
        mhdGlmFlux(gamma,ch,vord,bord,uR,F);
      }

      return fmax(fabs(sL),fabs(sR));
    }
      
    KOKKOS_INLINE_FUNCTION
    Real mhdGlmKEPES(Real gamma,Real ch,int vord[3],int bord[3],
		     const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		     Real F[NUM_VARS])
    {
      using MHD_GLM::SimVar;
      using Kokkos::Experimental::fmax;
      using Kokkos::Experimental::fabs;
      using Kokkos::Experimental::sqrt;
      
      Real sL,sR,sMax;
      
      // Wave speed estimates
      mhdGlmSpeed(gamma,vord[0],bord,uL,uR,ch,&sL,&sR);
      sMax = fmax(fabs(sL),fabs(sR));

      // Find far too many mean quantites
      Real rhoLn = logMean(uL[SimVar::dens],uR[SimVar::dens]);
      Real rhoAv = arithMean(uL[SimVar::dens],uR[SimVar::dens]);
      
      Real uAv = arithMean(uL[vord[0]]/uL[SimVar::dens],uR[vord[0]]/uR[SimVar::dens]);
      Real vAv = arithMean(uL[vord[1]]/uL[SimVar::dens],uR[vord[1]]/uR[SimVar::dens]);
      Real wAv = arithMean(uL[vord[2]]/uL[SimVar::dens],uR[vord[2]]/uR[SimVar::dens]);
      
      Real uuAv = arithMean(uL[vord[0]]*uL[vord[0]]/(uL[SimVar::dens]*uL[SimVar::dens]),
			    uR[vord[0]]*uR[vord[0]]/(uR[SimVar::dens]*uR[SimVar::dens]));
      Real vvAv = arithMean(uL[vord[1]]*uL[vord[1]]/(uL[SimVar::dens]*uL[SimVar::dens]),
			    uR[vord[1]]*uR[vord[1]]/(uR[SimVar::dens]*uR[SimVar::dens]));
      Real wwAv = arithMean(uL[vord[2]]*uL[vord[2]]/(uL[SimVar::dens]*uL[SimVar::dens]),
			    uR[vord[2]]*uR[vord[2]]/(uR[SimVar::dens]*uR[SimVar::dens]));
      
      Real bxAv = arithMean(uL[bord[0]],uR[bord[0]]);
      Real byAv = arithMean(uL[bord[1]],uR[bord[1]]);
      Real bzAv = arithMean(uL[bord[2]],uR[bord[2]]);
      
      Real bxxAv = arithMean(uL[bord[0]]*uL[bord[0]],uR[bord[0]]*uR[bord[0]]);
      Real byyAv = arithMean(uL[bord[1]]*uL[bord[1]],uR[bord[1]]*uR[bord[1]]);
      Real bzzAv = arithMean(uL[bord[2]]*uL[bord[2]],uR[bord[2]]*uR[bord[2]]);
      
      Real uBxAv = arithMean(uL[vord[0]]*uL[bord[0]]/uL[SimVar::dens],
			     uR[vord[0]]*uR[bord[0]]/uR[SimVar::dens]);
      Real vByAv = arithMean(uL[vord[1]]*uL[bord[1]]/uL[SimVar::dens],
			     uR[vord[1]]*uR[bord[1]]/uR[SimVar::dens]);
      Real wBzAv = arithMean(uL[vord[2]]*uL[bord[2]]/uL[SimVar::dens],
			     uR[vord[2]]*uR[bord[2]]/uR[SimVar::dens]);
      
      Real uBxxAv = arithMean(uL[vord[0]]*uL[bord[0]]*uL[bord[0]]/uL[SimVar::dens],
			      uR[vord[0]]*uR[bord[0]]*uR[bord[0]]/uR[SimVar::dens]);
      Real uByyAv = arithMean(uL[vord[0]]*uL[bord[1]]*uL[bord[1]]/uL[SimVar::dens],
			      uR[vord[0]]*uR[bord[1]]*uR[bord[1]]/uR[SimVar::dens]);
      Real uBzzAv = arithMean(uL[vord[0]]*uL[bord[2]]*uL[bord[2]]/uL[SimVar::dens],
			      uR[vord[0]]*uR[bord[2]]*uR[bord[2]]/uR[SimVar::dens]);
      
      Real psiAv = arithMean(uL[SimVar::psi],uR[SimVar::psi]);
      Real bxPsiAv = arithMean(uL[bord[0]]*uL[SimVar::psi],uR[bord[0]]*uR[SimVar::psi]);

      Real betaLn = logMean(uL[SimVar::dens]/(2.0*EOS<EquationType::MHD_GLM>::prsg(gamma,uL)),
			    uR[SimVar::dens]/(2.0*EOS<EquationType::MHD_GLM>::prsg(gamma,uR)));
      Real betaAv = arithMean(uL[SimVar::dens]/(2.0*EOS<EquationType::MHD_GLM>::prsg(gamma,uL)),
			      uR[SimVar::dens]/(2.0*EOS<EquationType::MHD_GLM>::prsg(gamma,uR)));
      Real pTil = rhoAv/(2.0*betaAv);
      Real pHatTot = pTil + 0.5*(bxxAv + byyAv + bzzAv);

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
	(1.0/(2.0*(gamma - 1)*betaLn) - 0.5*(uuAv + vvAv + wwAv))*F[SimVar::dens] +
	uAv*F[vord[0]] + vAv*F[vord[1]] + wAv*F[vord[2]] +
	bxAv*F[bord[0]] + byAv*F[bord[1]] + bzAv*F[bord[2]] +
	psiAv*F[SimVar::psi] -
	0.5*(uBxxAv + uByyAv + uBzzAv) +
	bxAv*(uBxAv + vByAv + wBzAv) -
	ch*bxPsiAv;

      // Add in local Lax-Friedrichs dissipation
      for (int nV=0; nV<NUM_VARS; nV++) {
	F[nV] -= 0.5*sMax*(uR[nV] - uL[nV]);
      }

      return sMax;
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_X(const Real gamma,const Real ch,
		       const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		       Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz};
      int bord[] = {MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz};
      return mhdGlmHLL(gamma,ch,vord,bord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_Y(const Real gamma,const Real ch,
		       const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		       Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx};
      int bord[] = {MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx};
      return mhdGlmHLL(gamma,ch,vord,bord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_HLL_Z(const Real gamma,const Real ch,
		       const Real uL[NUM_VARS],const Real uR[NUM_VARS],
		       Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy};
      int bord[] = {MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy};
      return mhdGlmHLL(gamma,ch,vord,bord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_KEPES_X(const Real gamma,const Real ch,
			 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
			 Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz};
      int bord[] = {MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz};
      return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_KEPES_Y(const Real gamma,const Real ch,
			 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
			 Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momy,MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx};
      int bord[] = {MHD_GLM::SimVar::magy,MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx};
      return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,F);
    }

    KOKKOS_INLINE_FUNCTION
    Real MHD_GLM_KEPES_Z(const Real gamma,const Real ch,
			 const Real uL[NUM_VARS],const Real uR[NUM_VARS],
			 Real F[NUM_VARS])
    {
      int vord[] = {MHD_GLM::SimVar::momz,MHD_GLM::SimVar::momx,MHD_GLM::SimVar::momy};
      int bord[] = {MHD_GLM::SimVar::magz,MHD_GLM::SimVar::magx,MHD_GLM::SimVar::magy};
      return mhdGlmKEPES(gamma,ch,vord,bord,uL,uR,F);
    }
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
