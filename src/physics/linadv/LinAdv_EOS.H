#ifndef LINADV_EOS_H_
#define LINADV_EOS_H_

#include <Kokkos_Core.hpp>

#include "../SimVar.H"
#include "../../Types.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"

namespace KFVM {

  namespace Physics {

    template<>
    struct EOS<EquationType::LinAdv>
    {
      KOKKOS_INLINE_FUNCTION
      static void cons2prim(const Real gam,const LinAdv::ConsVec U,LinAdv::PrimVec V)
      {
        (void) gam; (void) V; (void) U;
      }
      
      KOKKOS_INLINE_FUNCTION
      static void prim2cons(const Real gam,const LinAdv::PrimVec V,LinAdv::ConsVec U)
      {
        (void) gam; (void) V; (void) U;
      }      

      KOKKOS_INLINE_FUNCTION
      static Real weno(const LinAdv::ConsVec U,const FluidProperties& fProp)
      {
        // Weno is all or nothing for now
        return 1.0;
      }
      
      KOKKOS_INLINE_FUNCTION
      static Real positivity(const LinAdv::ConsVec U)
      {
        // Positivity unimportant
        (void) U;
        return 1.0;
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
