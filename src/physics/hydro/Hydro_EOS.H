#ifndef HYDRO_EOS_H_
#define HYDRO_EOS_H_

#include <Kokkos_Core.hpp>

#include "Hydro_SimVar.H"
#include "../../Types.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::Hydro>
    {
      KOKKOS_INLINE_FUNCTION
      static void cons2prim(const Real gam,const Hydro::ConsVec U,Hydro::PrimVec V)
      {
        using Hydro::SimVar;
        using Hydro::PrimVar;

        V[PrimVar::dens] = U[SimVar::dens];
        V[PrimVar::velx] = U[SimVar::momx]/U[SimVar::dens];
        V[PrimVar::vely] = U[SimVar::momy]/U[SimVar::dens];
        V[PrimVar::velz] = U[SimVar::momz]/U[SimVar::dens];
        V[PrimVar::pres] = pres(gam,U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static void prim2cons(const Real gam,const Hydro::PrimVec V,Hydro::ConsVec U)
      {
        using Hydro::SimVar;
        using Hydro::PrimVar;

        U[SimVar::dens] = V[PrimVar::dens];
        U[SimVar::momx] = V[PrimVar::dens]*V[PrimVar::velx];
        U[SimVar::momy] = V[PrimVar::dens]*V[PrimVar::vely];
        U[SimVar::momz] = V[PrimVar::dens]*V[PrimVar::velz];
        U[SimVar::etot] = V[PrimVar::pres]/(gam - Real(1.0)) +
          Real(0.5)*(V[PrimVar::velx]*V[PrimVar::velx] +
                     V[PrimVar::vely]*V[PrimVar::vely] +
                     V[PrimVar::velz]*V[PrimVar::velz]);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real ekin(const Hydro::ConsVec U)
      {
	using Hydro::SimVar;
        return Real(0.5)*(U[SimVar::momx]*U[SimVar::momx] +
			  U[SimVar::momy]*U[SimVar::momy] +
			  U[SimVar::momz]*U[SimVar::momz])/U[SimVar::dens];
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real eint(const Hydro::ConsVec U)
      {
	using Hydro::SimVar;
        return (U[SimVar::etot] - ekin(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real weno(const Hydro::ConsVec U,const FluidProperties& fProp)
      {
	using Hydro::SimVar;
        return pres(fProp.gamma,U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const Hydro::ConsVec U)
      {
	using Hydro::SimVar;
        using Kokkos::fmin;
        // Ensure that density and internal energy are positive
        return fmin(U[SimVar::dens],eint(U));
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real pres(const Real gamma,const Hydro::ConsVec U)
      {  
        return (gamma - Real(1.0))*eint(U);
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real soundspeed(const Real gamma,const Hydro::ConsVec U)
      {
	using Hydro::SimVar;
        using Kokkos::sqrt;
        return sqrt(gamma*pres(gamma,U)/U[SimVar::dens]);
      }
      
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
