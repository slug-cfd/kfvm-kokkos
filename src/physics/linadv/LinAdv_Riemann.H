#ifndef LINADV_RIEMANN_H_
#define LINADV_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../SimVar.H"
#include "LinAdv_EOS.H"
#include "../EquationTypes.H"
#include "../Riemann.H"

namespace KFVM {

  namespace Physics {
      
    KOKKOS_INLINE_FUNCTION
    Real linadvExact(Real gamma,int dir,
                     const LinAdv::ConsVec uL,
                     const LinAdv::ConsVec uR,
                     Real F[NUM_VARS])
    {
      using Kokkos::sin;
      using Kokkos::cos;
      using LinAdv::SimVar;
      using LinAdv::ConsVec;

      // Velocity on face
      const Real vx = cos(gamma),vy = sin(gamma);

      if (dir == 0) {
        F[0] = vx > 0.0 ? vx*uL[0] : vx*uR[0];
      } else if (dir == 1) {
        F[0] = vy > 0.0 ? vy*uL[0] : vy*uR[0];
      } else {
        F[0] = 0.0;
      }

      return 1.0;
    }

    template<RSType rst> struct LinAdvRS {};

    template<>
    struct LinAdvRS<RSType::EXACT>
    {
      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_X(const Real gamma,
                          const LinAdv::ConsVec uL,const LinAdv::ConsVec uR,
                          LinAdv::ConsVec F)
      {
        return linadvExact(gamma,0,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Y(const Real gamma,
                          const LinAdv::ConsVec uL,const LinAdv::ConsVec uR,
                          LinAdv::ConsVec F)
      {
        return linadvExact(gamma,1,uL,uR,F);
      }

      static
      KOKKOS_INLINE_FUNCTION
      Real RiemannSolve_Z(const Real gamma,
                          const LinAdv::ConsVec uL,const LinAdv::ConsVec uR,
                          LinAdv::ConsVec F)
      {
        return linadvExact(gamma,2,uL,uR,F);
      }
    };
    
  } // end namespace Physics
  
} // end namespace KFVM

#endif
