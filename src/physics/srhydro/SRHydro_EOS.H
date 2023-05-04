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
	const Real ei = eint(gam,V),h = Real(1.0) + gam*ei;
        const Real W = V[PrimVar::lorz];

	U[SimVar::dens] = V[PrimVar::rho]*W;
	U[SimVar::momx] = U[SimVar::dens]*W*h*V[PrimVar::velx];
	U[SimVar::momy] = U[SimVar::dens]*W*h*V[PrimVar::vely];
	U[SimVar::momz] = U[SimVar::dens]*W*h*V[PrimVar::velz];
	U[SimVar::tau] = U[SimVar::dens]*(W*h - Real(1.0)) - V[PrimVar::pres];
      }
      
      // Generate initial guess as center of bracket range
      // using eq. D.32 in Relativistic Hydrodynamics
      KOKKOS_INLINE_FUNCTION
      static Real presGuess(const Real gam,const SRHydro::ConsVec U)
      {
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	const Real S = sqrt(U[SimVar::momx]*U[SimVar::momx] +
			    U[SimVar::momy]*U[SimVar::momy] +
			    U[SimVar::momz]*U[SimVar::momz]);
	const Real k = S/(U[SimVar::tau] + U[SimVar::dens]);

	// Bracket is on z = Wv
	const Real zm = k/(Real(2.0)*sqrt(Real(1.0) - k*k/Real(4.0)));
	const Real zp = k/sqrt(Real(1.0) - k*k);
	const Real zc = (zm + zp)/Real(2.0);

	// This gives estimate of Lorenz factor and enthalpy
	const Real Wsq = Real(1.0) + zc*zc;

	// convert that to pressure and return
        const Real gRatio = gam/(gam - Real(1.0));
        return (U[SimVar::tau] + U[SimVar::dens])/(Wsq*gRatio - Real(1.0));
      }

      // Get pressure by NR process
      // Consult appendix D of Relativistic Hydrodynamics
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gam,const int maxIt,const SRHydro::ConsVec U)
      {
	using Kokkos::fabs;
	using Kokkos::fmax;
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	// Set tolerances
	const Real rtol = Real(1.e-10);

	// Pull out useful quantities
	const Real gamm = gam - Real(1.0);
	const Real D = U[SimVar::dens],T = U[SimVar::tau];
	const Real SS = (U[SimVar::momx]*U[SimVar::momx] +
			 U[SimVar::momy]*U[SimVar::momy] +
			 U[SimVar::momz]*U[SimVar::momz]);
	
	// Use bracket to get decent initial guess
	Real p = presGuess(gam,U);

	// Do NR iteration to tolerance
	for (int n=0; n<maxIt; n++) {
	  // get rho and internal energy from U and current p
	  const Real tpd = D + T + p;
	  const Real rt = sqrt(tpd*tpd - SS);
	  const Real rho = D*rt/tpd;
	  const Real ei = (rt - p*tpd/rt - D)/D;

	  // get p from this rho and eint
	  const Real phat = gamm*rho*ei;

	  // test p for convergence
	  if (rtol > fabs(p - phat)/fmax(p,phat)) {
	    break;
	  }

	  // update p
	  const Real drdp = D*SS/(rt*tpd*tpd);
	  const Real dedp = p*SS/(rho*tpd*(tpd*tpd - SS));
	  const Real dFdp = Real(1.0) - gamm*(ei*drdp + rho*dedp);

	  p -= (p - phat)/dFdp;
	}
	
        return p;
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real lorenz(const Real p,const SRHydro::ConsVec U)
      {
	using SRHydro::SimVar;
        using Kokkos::sqrt;

	const Real DhW = U[SimVar::tau] + p + U[SimVar::dens];
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
      static Real weno(const SRHydro::ConsVec U,const FluidProperties& fProp)
      {
	using SRHydro::SimVar;
        
        return pres(fProp.gamma,10,U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gam,const SRHydro::PrimVec V)
      {
	using SRHydro::PrimVar;
        using Kokkos::sqrt;

        const Real ei = eint(gam,V);
	const Real h = Real(1.0) + gam*ei;
        
        return sqrt(gam*V[PrimVar::pres]/(V[PrimVar::rho]*h));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const SRHydro::ConsVec U)
      {
        using Kokkos::fmin;
        using Kokkos::sqrt;
	using SRHydro::SimVar;

        // Calculate lower bound on tau
        const Real tauMin = sqrt(U[SimVar::dens]*U[SimVar::dens] +
                                 U[SimVar::momx]*U[SimVar::momx] +
                                 U[SimVar::momy]*U[SimVar::momy] +
                                 U[SimVar::momz]*U[SimVar::momz]) - U[SimVar::dens];

        // Note that the bound on tau also guarantees positive pressure
        // and subluminal velocities
        return fmin(U[SimVar::dens],U[SimVar::tau] - tauMin);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
