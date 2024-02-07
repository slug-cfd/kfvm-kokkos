#ifndef LINADV_EOS_H_
#define LINADV_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../SimVar.H"

namespace KFVM {

namespace Physics {

template <>
struct EOS<EquationType::LinAdv> {
  using ConsVar = typename SimVar<EquationType::LinAdv>::ConsVar;
  using ConsVec = typename SimVar<EquationType::LinAdv>::ConsVec;
  using PrimVec = typename SimVar<EquationType::LinAdv>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gam, const ConsVec U, PrimVec V) {
    (void)gam;
    (void)V;
    (void)U;
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const PrimVec V, ConsVec U) {
    (void)gam;
    (void)V;
    (void)U;
  }

  // These are not technically correct, but also aren't used
  KOKKOS_INLINE_FUNCTION
  static Real flowdirx(const ConsVec U) { return 0.0; }

  KOKKOS_INLINE_FUNCTION
  static Real flowdiry(const ConsVec U) { return 0.0; }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirz(const ConsVec U) { return 0.0; }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const ConsVec U, const ParameterStructs &eosParams) { return 0.0; }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const ConsVec U) {
    // Positivity unimportant
    (void)U;
    return 1.0;
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
