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
        return 0.5*(U[EulerAdiabatic::SimVar::momx]*U[EulerAdiabatic::SimVar::momx] +
		    U[EulerAdiabatic::SimVar::momy]*U[EulerAdiabatic::SimVar::momy] +
		    U[EulerAdiabatic::SimVar::momz]*U[EulerAdiabatic::SimVar::momz])/U[EulerAdiabatic::SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real U[NUM_VARS])
      {
        return (U[EulerAdiabatic::SimVar::etot] - ekin(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(Real gamma,const Real U[NUM_VARS])
      {  
        return (gamma - 1.0)*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(Real gamma,const Real U[NUM_VARS])
      {
        return Kokkos::Experimental::sqrt(gamma*pres(gamma,U)/U[EulerAdiabatic::SimVar::dens]);
      }
      
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
