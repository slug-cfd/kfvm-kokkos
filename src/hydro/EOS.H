#ifndef EOS_H_
#define EOS_H_

#include <Kokkos_Core.hpp>

#include "../SimVar.H"
#include "../Types.H"

namespace KFVM {

  namespace Hydro {

    namespace EulerAdiabatic {

      KOKKOS_INLINE_FUNCTION
      Real ekin(const Real U[NUM_VARS])
      {
        return 0.5*(U[SimVar::momx]*U[SimVar::momx] +
		    U[SimVar::momy]*U[SimVar::momy] +
		    U[SimVar::momz]*U[SimVar::momz])/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      Real eint(const Real U[NUM_VARS])
      {
        return (U[SimVar::etot] - ekin(U))/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      Real pres(Real gamma,const Real U[NUM_VARS])
      {  
        return (gamma - 1.0)*(U[SimVar::etot] - ekin(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      Real soundspeed(Real gamma,const Real U[NUM_VARS])
      {
        return Kokkos::Experimental::sqrt(gamma*pres(gam,U)/U[SimVar::dens]);
      }
      
    } // end namespace EulerAdiabatic
    
  } // end namespace Hydro
  
} // end namespace KFVM

#endif
