#ifndef MHD_GLM_EOS_H_
#define MHD_GLM_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../Types.H"
#include "MHD_GLM_SimVar.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::MHD_GLM>
    {
      // Raw array versions
      KOKKOS_INLINE_FUNCTION
      static Real ekin(const Real U[NUM_VARS])
      {
        return 0.5*(U[MHD_GLM::SimVar::momx]*U[MHD_GLM::SimVar::momx] +
		    U[MHD_GLM::SimVar::momy]*U[MHD_GLM::SimVar::momy] +
		    U[MHD_GLM::SimVar::momz]*U[MHD_GLM::SimVar::momz])/U[MHD_GLM::SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsb(const Real U[NUM_VARS])
      {  
        return (U[MHD_GLM::SimVar::magx]*U[MHD_GLM::SimVar::magx] +
                U[MHD_GLM::SimVar::magy]*U[MHD_GLM::SimVar::magy] +
                U[MHD_GLM::SimVar::magz]*U[MHD_GLM::SimVar::magz])/2.0;
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real U[NUM_VARS])
      {
        return (U[MHD_GLM::SimVar::etot] - prsb(U) - ekin(U) -
		0.5*U[MHD_GLM::SimVar::psi]*U[MHD_GLM::SimVar::psi]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsg(Real gamma,const Real U[NUM_VARS])
      {
        return (gamma - 1.0)*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(Real gamma,const Real U[NUM_VARS])
      {  
        return prsg(gamma,U) + prsb(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(Real gamma,const Real U[NUM_VARS])
      {
        return Kokkos::Experimental::sqrt(gamma*prsg(gamma,U)/U[MHD_GLM::SimVar::dens]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonic(Real gamma,const Real U[NUM_VARS],Real bl)
      {
	using MHD_GLM::SimVar;
	using Kokkos::Experimental::sqrt;
	
	Real asq = gamma*prsg(gamma,U)/U[SimVar::dens],bsq = 2.0*prsb(U);
	Real cfsq = 0.5*(asq + bsq + sqrt((asq + bsq)*(asq + bsq) - 4.0*asq*bl*bl));
	
        return sqrt(cfsq);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicX(Real gamma,const Real U[NUM_VARS])
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magx]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicY(Real gamma,const Real U[NUM_VARS])
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magy]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicZ(Real gamma,const Real U[NUM_VARS])
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magz]);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
