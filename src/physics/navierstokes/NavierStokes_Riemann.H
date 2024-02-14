#ifndef NAVIERSTOKES_RIEMANN_H_
#define NAVIERSTOKES_RIEMANN_H_

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Types.H"
#include "../EquationTypes.H"
#include "../Riemann.H"
#include "../SimVar.H"
#include "../hydro/Hydro_Riemann.H"
#include "NavierStokes_EOS.H"

namespace KFVM {

namespace Physics {

template <>
struct RSPhys<EquationType::NavierStokes> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using PrimVar = typename SimVar<EquationType::NavierStokes>::PrimVar;
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;
  using PrimVec = typename SimVar<EquationType::NavierStokes>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void flux(const EosParameters &eosParams, const int ord[3], const ConsVec U,
                   const PrimVec V, ConsVec F) {
    // Forward to hydro
    RSPhys<EquationType::Hydro>::flux(eosParams, ord, U, V, F);
  }

  // Wavespeed estimates
  // See Batten et. al. https://doi.org/10.1137/S1064827593260140
  KOKKOS_INLINE_FUNCTION
  static Real speeds(const EosParameters &eosParams, const int ord[3], const ConsVec uL,
                     const ConsVec uR, const PrimVec vL, const PrimVec vR, Real &sL,
                     Real &sR) {
    // Forward to hydro
    return RSPhys<EquationType::Hydro>::speeds(eosParams, ord, uL, uR, vL, vR, sL, sR);
  }
};

// This incorporates the low-mach fix from the HLLC+ method
// It does *not* do any comparisons with neighboring cells (fM variable),
// and does not incorporate the shear velocity fix for carbuncling
// see https://doi.org/10.1137/18M119032X
template <>
struct RiemannSolver<EquationType::NavierStokes, RSType::HLLC> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using PrimVar = typename SimVar<EquationType::NavierStokes>::PrimVar;
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;
  using PrimVec = typename SimVar<EquationType::NavierStokes>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static Real solve(const EosParameters &eosParams, const int dir, const ConsVec uL,
                    const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
    // and again, defer to hydro
    return RiemannSolver<EquationType::Hydro, RSType::HLLC>::solve(eosParams, dir, uL, uR,
                                                                   vL, vR, F);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
