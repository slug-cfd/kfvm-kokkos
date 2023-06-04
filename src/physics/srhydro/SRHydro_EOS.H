#ifndef SRHYDRO_EOS_H_
#define SRHYDRO_EOS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "../SimVar.H"
#include "../../Types.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::SRHydro>
    {
      // Construction of primitives is not trivial
      // -> Elevate this into EOS to ensure that the
      //    expensive part is only done once
      // Maximum iterations explicitly passed to allow
      // some calling methods to use loose approximations
      // when appropriate
      KOKKOS_INLINE_FUNCTION
      static void cons2prim(const Real gam,const int maxIt,
                            const SRHydro::ConsVec U,SRHydro::PrimVec V)
      {
        using SRHydro::SimVar;
        using SRHydro::PrimVar;
        
        // Single expensive solve can be done for pressure,
        // do it first and pass result to other EOS routines
        const Real p = pres(gam,maxIt,U);

        // Get remaining quantities cheaply
        const Real W = lorenz(p,U);
        const Real rho = U[SimVar::dens]/W,ei = eint(gam,p,U),h = Real(1.0) + gam*ei;
	
	// Get velocities and fill everything in
	V[PrimVar::rho] = rho;
	V[PrimVar::velx] = U[SimVar::momx]/(rho*W*W*h);
	V[PrimVar::vely] = U[SimVar::momy]/(rho*W*W*h);
	V[PrimVar::velz] = U[SimVar::momz]/(rho*W*W*h);
        V[PrimVar::lorz] = W;
	V[PrimVar::pres] = p;
      }

      // Conservative from primitive is easy
      KOKKOS_INLINE_FUNCTION
      static void prim2cons(const Real gam,const SRHydro::PrimVec V,SRHydro::ConsVec U)
      {
        using SRHydro::SimVar;
        using SRHydro::PrimVar;

	// Get enthalpy
	const Real h = enthalpy(gam,V);
        const Real W = V[PrimVar::lorz];

	U[SimVar::dens] = V[PrimVar::rho]*W;
	U[SimVar::momx] = U[SimVar::dens]*W*h*V[PrimVar::velx];
	U[SimVar::momy] = U[SimVar::dens]*W*h*V[PrimVar::vely];
	U[SimVar::momz] = U[SimVar::dens]*W*h*V[PrimVar::velz];
	U[SimVar::etot] = U[SimVar::dens]*W*h - V[PrimVar::pres];
      }

      // All pressure recovery methods come from https://doi.org/10.48550/arXiv.2305.14805
      // Initial guess given by eqs (2.21) and (2.16)
      KOKKOS_INLINE_FUNCTION
      static Real presGuess(const Real gam,const SRHydro::ConsVec U)
      {
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	const Real ESsq = U[SimVar::etot]*U[SimVar::etot] - (U[SimVar::momx]*U[SimVar::momx] +
							     U[SimVar::momy]*U[SimVar::momy] +
							     U[SimVar::momz]*U[SimVar::momz]);

	if (U[SimVar::dens] >= ESsq/U[SimVar::etot]) {
	  return 0.0;
	}

	const Real gm2E = (gam - Real(2.0))*U[SimVar::etot];
	const Real gm = gam - Real(1.0);
        return (gm2E + sqrt(gm2E*gm2E -
			    Real(4.0)*gm*(U[SimVar::dens]*sqrt(ESsq) - ESsq)))/Real(2.0);
      }

      // More stable scheme, algo (2.3)
      KOKKOS_INLINE_FUNCTION
      static Real presStable(const Real gam,const int maxIt,const Real rtol,const SRHydro::ConsVec U,Real p)
      {
	using Kokkos::fabs;
	using Kokkos::fmax;
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	// Useful quantities
	const Real gm = gam - Real(1.0);
	const Real gm2 = gam - Real(2.0);
	const Real D = U[SimVar::dens];
	const Real E = U[SimVar::etot];
	const Real Ssq = (U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz]);
	for (int n=0; n<maxIt; n++) {
	  // Recurring quantities
	  const Real EP = E + p;
	  const Real rt = sqrt(EP*EP - Ssq);
	  
	  // Evaluate psi and derivative
	  const Real psiP = Ssq + EP*(p/gm - E) + D*rt;
	  const Real dpsiP = (Real(2.0)*p - gm2*E)/gm + D*EP/rt;

	  // Update p and test for convergence
	  const Real dp = -psiP/dpsiP;
	  if (fabs(dp)/p < rtol) {
	    break;
	  }
	  p += dp;
	}
	
        return p;
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gam,const int maxIt,const SRHydro::ConsVec U)
      {
	using Kokkos::fabs;
	using Kokkos::fmax;
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	// Set tolerances
	const Real eps1 = Real(0.01),eps2 = Real(1.e-4);
	const Real rtol = Real(1.e-9);
	
	// Use bracket to get decent initial guess
	Real p = presGuess(gam,U);
	
	// Check which scheme to apply
	const Real Dsq = U[SimVar::dens]*U[SimVar::dens];
	const Real Esq = U[SimVar::etot]*U[SimVar::etot];
	const Real Ssq = (U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz]);
	const Real ESsq = Esq - Ssq;
	// Immediately swap to stable version
	if (Dsq/ESsq < eps2) {
	  return presStable(gam,maxIt,rtol,U,p);
	}

	// Do fast version unless lorenz factor is bad
	// Algorithm (2.1)
	const Real gm = gam - Real(1.0);
	const Real gm2 = gam - Real(2.0);
	const Real c0 = -ESsq*(Dsq - ESsq)*gm*gm;
	const Real c1 = Real(2.0)*U[SimVar::etot]*(gm2*ESsq*gm - Dsq*gm*gm);
	const Real c2 = Esq*(gam*gam - Real(6.0)*gm) + Real(2.0)*Ssq*gm - Dsq*gm*gm;
	const Real c3 = -Real(2.0)*U[SimVar::etot]*gm2;
	for (int n=0; n<maxIt; n++) {
	  // Test lorenz factor
	  // revert to stable if needed
	  const Real W = lorenz(p,U);
	  if (W < Real(1.0) + eps1) {
	    return presStable(gam,maxIt,rtol,U,p);
	  }
	  
	  // Evaluate pressure polynomial and derivative
	  const Real phiP = c0 + p*(c1 + p*(c2 + p*(c3 + p)));
	  const Real dphiP = c1 + p*(Real(2.0)*c2 + p*(Real(3.0)*c3 + p));

	  // Update p and test for convergence
	  const Real dp = -phiP/dphiP;
	  if (fabs(dp)/p < rtol) {
	    break;
	  }
	  p += dp;
	}
	
        return p;
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real lorenz(const Real p,const SRHydro::ConsVec U)
      {
	using SRHydro::SimVar;
        using Kokkos::sqrt;

	const Real DhW = U[SimVar::etot] + p;
        const Real DhWsq = DhW*DhW;
        const Real msq =
          U[SimVar::momx]*U[SimVar::momx] +
          U[SimVar::momy]*U[SimVar::momy] +
          U[SimVar::momz]*U[SimVar::momz];
        return Real(1.0)/sqrt(Real(1.0) - msq/DhWsq);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real gam,const Real p,const SRHydro::ConsVec U)
      {
	using SRHydro::SimVar;
        
        const Real rho = U[SimVar::dens]/lorenz(p,U);
        return p/((gam - Real(1.0))*rho);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real gam,const SRHydro::PrimVec V)
      {
	using SRHydro::PrimVar;
        
        return V[PrimVar::pres]/((gam - Real(1.0))*V[PrimVar::rho]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real enthalpy(const Real gam,const Real p,const SRHydro::ConsVec U)
      {
	using SRHydro::SimVar;
        
        const Real rho = U[SimVar::dens]/lorenz(p,U);
        return Real(1.0) + gam*p/((gam - Real(1.0))*rho);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real enthalpy(const Real gam,const SRHydro::PrimVec V)
      {
	using SRHydro::PrimVar;
        
        return Real(1.0) + gam*V[PrimVar::pres]/((gam - Real(1.0))*V[PrimVar::rho]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real weno(const SRHydro::ConsVec U,const FluidProperties& fProp)
      {
	using SRHydro::SimVar;
        
        return U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gam,const SRHydro::PrimVec V)
      {
	using SRHydro::PrimVar;
        using Kokkos::sqrt;

	const Real h = enthalpy(gam,V);
        
        return sqrt(gam*V[PrimVar::pres]/(V[PrimVar::rho]*h));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const SRHydro::ConsVec U)
      {
        using Kokkos::fmin;
	using SRHydro::SimVar;

        // Calculate lower bound on etot
        const Real etotDiff = U[SimVar::etot]*U[SimVar::etot] - (U[SimVar::dens]*U[SimVar::dens] + 
                                                                 U[SimVar::momx]*U[SimVar::momx] +
                                                                 U[SimVar::momy]*U[SimVar::momy] +
                                                                 U[SimVar::momz]*U[SimVar::momz]);
        
        // Note that the bound on etot also guarantees positive pressure
        // and subluminal velocities
        return fmin(U[SimVar::dens],etotDiff);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
