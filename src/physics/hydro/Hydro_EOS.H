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
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::Hydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gam, const ConsVec U, PrimVec V) {
    V[PrimVar::dens] = U[ConsVar::dens];
    V[PrimVar::velx] = U[ConsVar::momx] / U[ConsVar::dens];
    V[PrimVar::vely] = U[ConsVar::momy] / U[ConsVar::dens];
    V[PrimVar::velz] = U[ConsVar::momz] / U[ConsVar::dens];
    V[PrimVar::pres] = pres(gam, U);
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const PrimVec V, ConsVec U) {
    U[ConsVar::dens] = V[PrimVar::dens];
    U[ConsVar::momx] = V[PrimVar::dens] * V[PrimVar::velx];
    U[ConsVar::momy] = V[PrimVar::dens] * V[PrimVar::vely];
    U[ConsVar::momz] = V[PrimVar::dens] * V[PrimVar::velz];
    U[ConsVar::etot] = V[PrimVar::pres] / (gam - Real(1.0)) +
                       Real(0.5) * (V[PrimVar::velx] * V[PrimVar::velx] +
                                    V[PrimVar::vely] * V[PrimVar::vely] +
                                    V[PrimVar::velz] * V[PrimVar::velz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const ConsVec U) {
    return Real(0.5) *
           (U[ConsVar::momx] * U[ConsVar::momx] + U[ConsVar::momy] * U[ConsVar::momy] +
            U[ConsVar::momz] * U[ConsVar::momz]) /
           U[ConsVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const ConsVec U) { return (U[ConsVar::etot] - ekin(U)); }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirx(const ConsVec U) {
    return U[ConsVar::momx] > 0.0 ? 1.0 : (U[ConsVar::momx] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdiry(const ConsVec U) {
    return U[ConsVar::momy] > 0.0 ? 1.0 : (U[ConsVar::momy] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirz(const ConsVec U) {
    return U[ConsVar::momz] > 0.0 ? 1.0 : (U[ConsVar::momz] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const ConsVec U, const EosParameters &eosParams) {
    using Kokkos::pow;

    return pres(eosParams.gamma, U) / pow(U[ConsVar::dens], eosParams.gamma);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const ConsVec U) {
    using Kokkos::fmin;
    // Ensure that density and internal energy are positive
    return fmin(U[ConsVar::dens], eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real pres(const Real gamma, const ConsVec U) {
    return (gamma - Real(1.0)) * eint(U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const ConsVec U) {
    using Kokkos::sqrt;
    return sqrt(gamma * pres(gamma, U) / U[ConsVar::dens]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
