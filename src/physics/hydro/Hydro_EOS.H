#ifndef HYDRO_EOS_H_
#define HYDRO_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../SimVar.H"

namespace KFVM {

namespace Physics {

template <>
struct EOS<EquationType::Hydro> {
  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gam, const Hydro::ConsVec U, Hydro::PrimVec V) {
    using Hydro::PrimVar;
    using Hydro::SimVar;

    V[PrimVar::dens] = U[SimVar::dens];
    V[PrimVar::velx] = U[SimVar::momx] / U[SimVar::dens];
    V[PrimVar::vely] = U[SimVar::momy] / U[SimVar::dens];
    V[PrimVar::velz] = U[SimVar::momz] / U[SimVar::dens];
    V[PrimVar::pres] = pres(gam, U);
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const Hydro::PrimVec V, Hydro::ConsVec U) {
    using Hydro::PrimVar;
    using Hydro::SimVar;

    U[SimVar::dens] = V[PrimVar::dens];
    U[SimVar::momx] = V[PrimVar::dens] * V[PrimVar::velx];
    U[SimVar::momy] = V[PrimVar::dens] * V[PrimVar::vely];
    U[SimVar::momz] = V[PrimVar::dens] * V[PrimVar::velz];
    U[SimVar::etot] = V[PrimVar::pres] / (gam - Real(1.0)) +
                      Real(0.5) * (V[PrimVar::velx] * V[PrimVar::velx] +
                                   V[PrimVar::vely] * V[PrimVar::vely] +
                                   V[PrimVar::velz] * V[PrimVar::velz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return Real(0.5) *
           (U[SimVar::momx] * U[SimVar::momx] + U[SimVar::momy] * U[SimVar::momy] +
            U[SimVar::momz] * U[SimVar::momz]) /
           U[SimVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return (U[SimVar::etot] - ekin(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirx(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return U[SimVar::momx] > 0.0 ? 1.0 : (U[SimVar::momx] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdiry(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return U[SimVar::momy] > 0.0 ? 1.0 : (U[SimVar::momy] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirz(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return U[SimVar::momz] > 0.0 ? 1.0 : (U[SimVar::momz] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const Hydro::ConsVec U, const EosParameters &eosParams) {
    using Hydro::SimVar;
    // using Kokkos::log;
    using Kokkos::pow;

    return pres(eosParams.gamma, U) / pow(U[SimVar::dens], eosParams.gamma);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const Hydro::ConsVec U) {
    using Hydro::SimVar;
    using Kokkos::fmin;
    // Ensure that density and internal energy are positive
    return fmin(U[SimVar::dens], eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real pres(const Real gamma, const Hydro::ConsVec U) {
    return (gamma - Real(1.0)) * eint(U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    using Kokkos::sqrt;
    return sqrt(gamma * pres(gamma, U) / U[SimVar::dens]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
