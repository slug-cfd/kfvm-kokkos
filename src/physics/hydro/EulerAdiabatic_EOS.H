#ifndef EULERADIABATIC_EOS_H_
#define EULERADIABATIC_EOS_H_

#include <Kokkos_Core.hpp>

#include "EulerAdiabatic_SimVar.H"
#include "../../Types.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::Euler>
    {
      KOKKOS_INLINE_FUNCTION
      static Real ekin(const Real U[NUM_VARS])
      {
	using EulerAdiabatic::SimVar;
        return Real(0.5)*(U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz])/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real U[NUM_VARS])
      {
	using EulerAdiabatic::SimVar;
        return (U[SimVar::etot] - ekin(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const Real U[NUM_VARS])
      {
	using EulerAdiabatic::SimVar;
        using Kokkos::fmin;
        // Ensure that density and internal energy are positive
        return fmin(U[SimVar::dens],eint(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(Real gamma,const Real U[NUM_VARS])
      {  
        return (gamma - Real(1.0))*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(Real gamma,const Real U[NUM_VARS])
      {
	using EulerAdiabatic::SimVar;
        using Kokkos::sqrt;
        return sqrt(gamma*pres(gamma,U)/U[SimVar::dens]);
      }
      
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
