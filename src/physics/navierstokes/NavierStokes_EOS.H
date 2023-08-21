#ifndef NAVIERSTOKES_EOS_H_
#define NAVIERSTOKES_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../SimVar.H"

namespace KFVM {

namespace Physics {

template <>
struct EOS<EquationType::NavierStokes> {
  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gamma, const NavierStokes::ConsVec U,
                        NavierStokes::PrimVec V) {
    using NavierStokes::PrimVar;
    using NavierStokes::SimVar;

    V[PrimVar::dens] = U[SimVar::dens];
    V[PrimVar::velx] = U[SimVar::momx] / U[SimVar::dens];
    V[PrimVar::vely] = U[SimVar::momy] / U[SimVar::dens];
    V[PrimVar::velz] = U[SimVar::momz] / U[SimVar::dens];
    V[PrimVar::pres] = pres(gamma, U);
    V[PrimVar::temp] = temp(gamma, U);
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gamma, const NavierStokes::PrimVec V,
                        NavierStokes::ConsVec U) {
    using NavierStokes::PrimVar;
    using NavierStokes::SimVar;

    const Real pd = (gamma - 1.0);

    U[SimVar::dens] = V[PrimVar::dens];
    U[SimVar::momx] = V[PrimVar::dens] * V[PrimVar::velx];
    U[SimVar::momy] = V[PrimVar::dens] * V[PrimVar::vely];
    U[SimVar::momz] = V[PrimVar::dens] * V[PrimVar::velz];
    U[SimVar::etot] =
        V[PrimVar::pres] / pd + Real(0.5) * (V[PrimVar::velx] * V[PrimVar::velx] +
                                             V[PrimVar::vely] * V[PrimVar::vely] +
                                             V[PrimVar::velz] * V[PrimVar::velz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return Real(0.5) *
           (U[SimVar::momx] * U[SimVar::momx] + U[SimVar::momy] * U[SimVar::momy] +
            U[SimVar::momz] * U[SimVar::momz]) /
           U[SimVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return (U[SimVar::etot] - ekin(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirx(const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return U[SimVar::momx] > 0.0 ? 1.0 : (U[SimVar::momx] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdiry(const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return U[SimVar::momy] > 0.0 ? 1.0 : (U[SimVar::momy] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirz(const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return U[SimVar::momz] > 0.0 ? 1.0 : (U[SimVar::momz] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const NavierStokes::ConsVec U, const EosParameters &eosParams) {
    using Kokkos::pow;
    using NavierStokes::SimVar;

    return pres(eosParams.gamma, U) / pow(U[SimVar::dens], eosParams.gamma);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const NavierStokes::ConsVec U) {
    using Kokkos::fmin;
    using NavierStokes::SimVar;
    // Ensure that density and internal energy are positive
    return fmin(U[SimVar::dens], eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real pres(const Real gamma, const NavierStokes::ConsVec U) {
    return (gamma - 1.0) * eint(U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real temp(const Real gamma, const NavierStokes::ConsVec U) {
    using NavierStokes::SimVar;
    return pres(gamma, U) / U[SimVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const NavierStokes::ConsVec U) {
    using Kokkos::sqrt;
    using NavierStokes::SimVar;
    return sqrt(gamma * pres(gamma, U) / U[SimVar::dens]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
