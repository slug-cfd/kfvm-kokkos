#ifndef MHD_GLM_EOS_H_
#define MHD_GLM_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../Types.H"
#include "../../FluidProperties.H"
#include "MHD_GLM_SimVar.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::MHD_GLM>
    {
      KOKKOS_INLINE_FUNCTION
      static void cons2prim(const Real gam,const MHD_GLM::ConsVec U,MHD_GLM::PrimVec V)
      {
        using MHD_GLM::SimVar;
        using MHD_GLM::PrimVar;

        V[PrimVar::dens] = U[SimVar::dens];
        V[PrimVar::velx] = U[SimVar::momx]/U[SimVar::dens];
        V[PrimVar::vely] = U[SimVar::momy]/U[SimVar::dens];
        V[PrimVar::velz] = U[SimVar::momz]/U[SimVar::dens];
        V[PrimVar::pres] = pres(gam,U);
        V[PrimVar::magx] = U[SimVar::magx];
        V[PrimVar::magy] = U[SimVar::magy];
        V[PrimVar::magz] = U[SimVar::magz];
        V[PrimVar::psi] = U[SimVar::psi];
      }
      
      KOKKOS_INLINE_FUNCTION
      static void prim2cons(const Real gam,const MHD_GLM::PrimVec V,MHD_GLM::ConsVec U)
      {
        using MHD_GLM::SimVar;
        using MHD_GLM::PrimVar;

        U[SimVar::dens] = V[PrimVar::dens];
        U[SimVar::momx] = V[PrimVar::dens]*V[PrimVar::velx];
        U[SimVar::momy] = V[PrimVar::dens]*V[PrimVar::vely];
        U[SimVar::momz] = V[PrimVar::dens]*V[PrimVar::velz];
        U[SimVar::etot] = V[PrimVar::pres]/(gam - Real(1.0)) +
          Real(0.5)*(V[PrimVar::velx]*V[PrimVar::velx] +
                     V[PrimVar::vely]*V[PrimVar::vely] +
                     V[PrimVar::velz]*V[PrimVar::velz]);
        U[SimVar::magx] = V[PrimVar::magx];
        U[SimVar::magy] = V[PrimVar::magy];
        U[SimVar::magz] = V[PrimVar::magz];
        U[SimVar::psi] = V[PrimVar::psi];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real ekin(const MHD_GLM::ConsVec U)
      {
        using MHD_GLM::SimVar;
        return Real(0.5)*(U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz])/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsb(const MHD_GLM::ConsVec U)
      {
	using MHD_GLM::SimVar;
        return Real(0.5)*(U[SimVar::magx]*U[SimVar::magx] +
			  U[SimVar::magy]*U[SimVar::magy] +
			  U[SimVar::magz]*U[SimVar::magz]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const MHD_GLM::ConsVec U)
      {
	using MHD_GLM::SimVar;
        return (U[SimVar::etot] - prsb(U) - ekin(U) - Real(0.5)*U[SimVar::psi]*U[SimVar::psi]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real weno(const MHD_GLM::ConsVec U,const FluidProperties& fProp)
      {
	using MHD_GLM::SimVar;
        return prsg(fProp.gamma,U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const MHD_GLM::ConsVec U)
      {
	using Kokkos::fmin;
	using MHD_GLM::SimVar;
        return fmin(U[SimVar::dens],eint(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real prsg(const Real gamma,const MHD_GLM::ConsVec U)
      {
        return (gamma - Real(1.0))*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gamma,const MHD_GLM::ConsVec U)
      {  
        return prsg(gamma,U) + prsb(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gamma,const MHD_GLM::ConsVec U)
      {
	using MHD_GLM::SimVar;
        return Kokkos::sqrt(gamma*prsg(gamma,U)/U[SimVar::dens]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonic(const Real gamma,const MHD_GLM::ConsVec U,Real bl)
      {
	using MHD_GLM::SimVar;
	using Kokkos::sqrt;
	
	Real asq = gamma*prsg(gamma,U)/U[SimVar::dens],bsq = Real(2.0)*prsb(U);
	Real cfsq = Real(0.5)*(asq + bsq + sqrt((asq + bsq)*(asq + bsq) - Real(4.0)*asq*bl*bl));
	
        return sqrt(cfsq);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicX(const Real gamma,const MHD_GLM::ConsVec U)
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magx]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicY(const Real gamma,const MHD_GLM::ConsVec U)
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magy]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real fastMagSonicZ(const Real gamma,const MHD_GLM::ConsVec U)
      {
        return fastMagSonic(gamma,U,U[MHD_GLM::SimVar::magz]);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
