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
  static void cons2prim(const Real gamma, const Real J, const Hydro::ConsVec U,
                        Hydro::PrimVec V) {
    using Hydro::PrimVar;
    using Hydro::SimVar;

    V[PrimVar::dens] = U[SimVar::dens] / J;
    V[PrimVar::velx] = U[SimVar::momx] / U[SimVar::dens];
    V[PrimVar::vely] = U[SimVar::momy] / U[SimVar::dens];
    V[PrimVar::velz] = U[SimVar::momz] / U[SimVar::dens];
    V[PrimVar::pres] = pres(gamma, J, U);
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gamma, const Real J, const Hydro::PrimVec V,
                        Hydro::ConsVec U) {
    using Hydro::PrimVar;
    using Hydro::SimVar;

    U[SimVar::dens] = J * V[PrimVar::dens];
    U[SimVar::momx] = J * V[PrimVar::dens] * V[PrimVar::velx];
    U[SimVar::momy] = J * V[PrimVar::dens] * V[PrimVar::vely];
    U[SimVar::momz] = J * V[PrimVar::dens] * V[PrimVar::velz];
    U[SimVar::etot] = J * (V[PrimVar::pres] / (gamma - 1.0) +
                           0.5 * V[PrimVar::dens] *
                               (V[PrimVar::velx] * V[PrimVar::velx] +
                                V[PrimVar::vely] * V[PrimVar::vely] +
                                V[PrimVar::velz] * V[PrimVar::velz]));
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const Real J, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    const Real rho = U[SimVar::dens] / J;
    const Real vx = U[SimVar::momx] / U[SimVar::dens];
    const Real vy = U[SimVar::momy] / U[SimVar::dens];
    const Real vz = U[SimVar::momz] / U[SimVar::dens];
    return 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const Real J, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return (U[SimVar::etot] / J - ekin(J, U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const EosParameters &eosParams, const Real J, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    return pres(eosParams.gamma, J, U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const Real J, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    using Kokkos::fmin;
    // Ensure that density and internal energy are positive
    return fmin(U[SimVar::dens], eint(J, U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real pres(const Real gamma, const Real J, const Hydro::ConsVec U) {
    return (gamma - 1.0) * eint(J, U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const Real J, const Hydro::ConsVec U) {
    using Hydro::SimVar;
    using Kokkos::sqrt;
    return sqrt(gamma * pres(gamma, J, U) * J / U[SimVar::dens]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
