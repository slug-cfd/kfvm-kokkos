#ifndef SRHYDRO_EOS_H_
#define SRHYDRO_EOS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "SRHydro_SimVar.H"
#include "../../Types.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::SRHydro>
    {
      // Generate initial guess as center of bracket range
      // using eq. D.32 in Relativistic Hydrodynamics
      KOKKOS_INLINE_FUNCTION
      static Real presGuess(const Real U[NUM_VARS])
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
	const Real W = sqrt(Real(1.0) + zc*zc);

	// convert that to pressure and return
        return S*S/(W*W - Real(1.0)) - U[SimVar::tau] - U[SimVar::dens];
      }

      // Get pressure by NR process
      // Consult appendix D of Relativistic Hydrodynamics
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gam,const Real U[NUM_VARS])
      {
	using Kokkos::fabs;
	using Kokkos::sqrt;
	using SRHydro::SimVar;

	// Set tolerances
	const Real rtol = Real(1.e-5),atol = Real(1.e-12);

	// Pull out useful quantities
	const Real gamm = gam - Real(1.0);
	const Real D = U[SimVar::dens],T = U[SimVar::tau];
	const Real SS = (U[SimVar::momx]*U[SimVar::momx] +
			 U[SimVar::momy]*U[SimVar::momy] +
			 U[SimVar::momz]*U[SimVar::momz]);
	
	// Use bracket to get decent initial guess
	Real p = presGuess(U);

	// Do NR iteration to tolerance
	for (int n=0; n<5; n++) {
	  // get rho and internal energy from U and current p
	  const Real tpd = D + T + p;
	  const Real rt = sqrt(tpd*tpd - SS);
	  const Real rho = D*rt/tpd;
	  const Real ei = (rt - p*tpd/rt - D)/D;

	  // get p from this rho and eint
	  const Real phat = gamm*rho*ei;

	  // test p for convergence
	  if (rtol > (atol + fabs(p - phat))) {
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
      static Real lorenz(const Real p,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::sqrt;

	const Real etot = U[SimVar::tau] + U[SimVar::dens];
        const Real msq =
          U[SimVar::momx]*U[SimVar::momx] +
          U[SimVar::momy]*U[SimVar::momy] +
          U[SimVar::momz]*U[SimVar::momz];
        const Real epsq = (etot + p)*(etot + p);
        return Real(1.0)/sqrt(Real(1.0) + msq/epsq);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real weno(const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        return pres(Real(4.0/3.0),U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real gam,const Real p,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        const Real rho = U[SimVar::dens]/lorenz(p,U);
        return p/((gam - Real(1.0))*rho);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real gam,const Real V[NUM_AUX])
      {
	using SRHydro::AuxVar;
        return V[AuxVar::pres]/((gam - Real(1.0))*V[AuxVar::rho]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gam,const Real p,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::sqrt;

        const Real W = lorenz(p,U),rho = U[SimVar::dens]/W;
        const Real ei = eint(gam,p,U),h = Real(1.0) + gam*ei;
        
        return sqrt(gam*p/(rho*h));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gam,const Real V[NUM_AUX])
      {
	using SRHydro::AuxVar;
        using Kokkos::sqrt;

        const Real ei = eint(gam,V);
	const Real h = Real(1.0) + gam*ei;
        
        return sqrt(gam*V[AuxVar::pres]/(V[AuxVar::rho]*h));
      }

      // Construction of primitives is not trivial
      // -> Elevate this into EOS to ensure that the
      //    expensive part is only done once
      KOKKOS_INLINE_FUNCTION
      static void primvec(const Real gam,const Real U[NUM_VARS],Real V[NUM_AUX])
      {
        using SRHydro::SimVar;
        using SRHydro::AuxVar;
        
        // Single expensive solve can be done for pressure,
        // do it first and pass result to other EOS routines
        const Real p = pres(gam,U);

        // Get remaining quantities cheaply
        const Real W = lorenz(p,U),rho = U[SimVar::dens]/W;
        const Real ei = eint(gam,p,U),h = Real(1.0) + gam*ei;
	
	// Get velocities and fill everything in
	V[AuxVar::rho] = rho;
	V[AuxVar::lorenz] = W;
	V[AuxVar::velx] = U[SimVar::momx]/(rho*W*W*h);
	V[AuxVar::vely] = U[SimVar::momy]/(rho*W*W*h);
	V[AuxVar::velz] = U[SimVar::momz]/(rho*W*W*h);
	V[AuxVar::pres] = p;
      }

      // Conservative from primitive is easy but tedious
      KOKKOS_INLINE_FUNCTION
      static void consvec(const Real gam,const Real V[NUM_AUX],Real U[NUM_VARS])
      {
        using SRHydro::SimVar;
        using SRHydro::AuxVar;

	// unpack slightly for clarity and get enthalpy
	const Real rho = V[AuxVar::rho],W = V[AuxVar::lorenz],p = V[AuxVar::pres];
	const Real ei = eint(gam,V),h = Real(1.0) + gam*ei;

	U[SimVar::dens] = rho*W;
	U[SimVar::momx] = rho*W*W*h*V[AuxVar::velx];
	U[SimVar::momy] = rho*W*W*h*V[AuxVar::vely];
	U[SimVar::momz] = rho*W*W*h*V[AuxVar::velz];
	U[SimVar::tau] = rho*W*W*h - p - U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::fmin;
        
        // Ensure that density and pressure are positive
        // Note that this interface does *not* have access to gamma
        // however, we can use any placeholder gamma here regardless
        // (minor hack, and can we really?)
        const Real p = pres(Real(4.0)/Real(3.0),U);

        // This should also check that velocities are bounded by 1...
        return fmin(U[SimVar::dens],p);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
