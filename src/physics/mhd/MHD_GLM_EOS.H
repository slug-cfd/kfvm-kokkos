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
        using MHD_GLM::SimVar;
        return Real(0.5)*(U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz])/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsb(const Real U[NUM_VARS])
      {
	using MHD_GLM::SimVar;
        return Real(0.5)*(U[SimVar::magx]*U[SimVar::magx] +
			  U[SimVar::magy]*U[SimVar::magy] +
			  U[SimVar::magz]*U[SimVar::magz]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Real U[NUM_VARS])
      {
	using MHD_GLM::SimVar;
        return (U[SimVar::etot] - prsb(U) - ekin(U) - Real(0.5)*U[SimVar::psi]*U[SimVar::psi]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real weno(const Real U[NUM_VARS])
      {
	using MHD_GLM::SimVar;
        return eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const Real U[NUM_VARS])
      {
	using Kokkos::Experimental::fmin;
	using MHD_GLM::SimVar;
        return fmin(U[SimVar::dens],eint(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsg(Real gamma,const Real U[NUM_VARS])
      {
        return (gamma - Real(1.0))*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(Real gamma,const Real U[NUM_VARS])
      {  
        return prsg(gamma,U) + prsb(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(Real gamma,const Real U[NUM_VARS])
      {
	using MHD_GLM::SimVar;
        return Kokkos::Experimental::sqrt(gamma*prsg(gamma,U)/U[SimVar::dens]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonic(Real gamma,const Real U[NUM_VARS],Real bl)
      {
	using MHD_GLM::SimVar;
	using Kokkos::Experimental::sqrt;
	
	Real asq = gamma*prsg(gamma,U)/U[SimVar::dens],bsq = Real(2.0)*prsb(U);
	Real cfsq = Real(0.5)*(asq + bsq + sqrt((asq + bsq)*(asq + bsq) - Real(4.0)*asq*bl*bl));
	
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
