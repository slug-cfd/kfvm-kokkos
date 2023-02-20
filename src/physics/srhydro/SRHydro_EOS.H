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
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gamma,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        return Real(1.0);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real lorenz(const Real pres,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::Experimental::sqrt;

        const Real msq =
          U[SimVar::momx]*U[SimVar::momx] +
          U[SimVar::momy]*U[SimVar::momy] +
          U[SimVar::momz]*U[SimVar::momz];
        const Real epsq = (U[SimVar::etot] + p)*(U[SimVar::etot] + p);
        return Real(1.0)/sqrt(Real(1.0) + msq/epsq);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real gamma,const Real pres,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        const Real rho = U[SimVar::dens]/lorenz(pres,U);
        return pres/((gamma - Real(1.0))*rho);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gamma,const Real pres,const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::Experimental::sqrt;

        const Real W = lorenz(p,ua),rho = U[SimVar::dens]/W;
        const Real ei = eint(fProp.gamma,p,ua),h = Real(1.0) + ei + p/rho;
        
        return sqrt(gamma*p/(rho*h));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const Real U[NUM_VARS])
      {
	using SRHydro::SimVar;
        using Kokkos::Experimental::fmin;
        
        // Ensure that density and pressure are positive
        // Note that this interface does *not* have access to gamma
        // however, we can use any placeholder gamma here regardless
        // (minor hack, and can we really?)
        const Real p = pres(Real(4.0)/Real(3.0),U);

        // This should also check that velocities are bounded by 1...
        return fmin(U[SimVar::dens],p);
      }

      // Construction of primitives is not trivial
      // -> Elevate this into EOS to ensure that the
      //    expensive part is only done once
      KOKKOS_INLINE_FUNCTION
      static void primvec(const Real gamma,const Real U[NUM_VARS],Real V[NUM_VARS])
      {
        using SRHydro::SimVar;
        using SRHydro::AuxVar;
        
        // Single expensive solve can be done for pressure,
        // do it first and pass result to other EOS routines
        const Real p = pres(fProp.gamma,ua);

        // Get remaining quantities cheaply
        const Real W = lorenz(p,ua),rho = U[SimVar::dens]/W;
        const Real ei = eint(fProp.gamma,p,ua),h = Real(1.0) + ei + p/rho;
	
	// Get velocities and fill everything in
	V[AuxVar::rho] = rho;
	V[AuxVar::lorenz] = W;
	V[AuxVar::velx] = U[SimVar::momx]/(rho*W*W*h);
	V[AuxVar::vely] = U[SimVar::momy]/(rho*W*W*h);
	V[AuxVar::velz] = U[SimVar::momz]/(rho*W*W*h);
	V[AuxVar::pres] = p;
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
