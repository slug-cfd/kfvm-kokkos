#ifndef RIEMANN_H_
#define RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"

namespace KFVM {

  namespace Physics {

    KOKKOS_INLINE_FUNCTION
    Real hll(Real sL,Real sR,Real fL,Real fR,Real uL,Real uR)
    {
      return ((sR*fL - sL*fR) + sL*sR*(uR - uL))/(sR - sL);
    }

    KOKKOS_INLINE_FUNCTION
    Real hllState(Real sL,Real sR,Real fL,Real fR,Real uL,Real uR)
    {
      return ((sR*uR - sL*uL) + (fL - fR))/(sR - sL);
    }
    
  } // end namespace Physics

} // end namespace KFVM

#endif
