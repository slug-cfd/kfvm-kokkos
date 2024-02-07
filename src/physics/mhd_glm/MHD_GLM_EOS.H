#ifndef MHD_GLM_EOS_H_
#define MHD_GLM_EOS_H_

#include <Kokkos_Core.hpp>

#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../SimVar.H"

namespace KFVM {

namespace Physics {

template <>
struct EOS<EquationType::MHD_GLM> {
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;
  using PrimVar = typename SimVar<EquationType::MHD_GLM>::PrimVar;
  using ConsVec = typename SimVar<EquationType::MHD_GLM>::ConsVec;
  using PrimVec = typename SimVar<EquationType::MHD_GLM>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gam, const ConsVec U, PrimVec V) {
    V[PrimVar::dens] = U[ConsVar::dens];
    V[PrimVar::velx] = U[ConsVar::momx] / U[ConsVar::dens];
    V[PrimVar::vely] = U[ConsVar::momy] / U[ConsVar::dens];
    V[PrimVar::velz] = U[ConsVar::momz] / U[ConsVar::dens];
    V[PrimVar::prsg] = prsg(gam, U);
    V[PrimVar::magx] = U[ConsVar::magx];
    V[PrimVar::magy] = U[ConsVar::magy];
    V[PrimVar::magz] = U[ConsVar::magz];
    V[PrimVar::psi] = U[ConsVar::psi];
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const PrimVec V, ConsVec U) {
    U[ConsVar::dens] = V[PrimVar::dens];
    U[ConsVar::momx] = V[PrimVar::dens] * V[PrimVar::velx];
    U[ConsVar::momy] = V[PrimVar::dens] * V[PrimVar::vely];
    U[ConsVar::momz] = V[PrimVar::dens] * V[PrimVar::velz];
    U[ConsVar::etot] =
        V[PrimVar::prsg] / (gam - Real(1.0)) +
        Real(0.5) *
            (V[PrimVar::dens] * (V[PrimVar::velx] * V[PrimVar::velx] +
                                 V[PrimVar::vely] * V[PrimVar::vely] +
                                 V[PrimVar::velz] * V[PrimVar::velz]) +
             V[PrimVar::magx] * V[PrimVar::magx] + V[PrimVar::magy] * V[PrimVar::magy] +
             V[PrimVar::magz] * V[PrimVar::magz] + V[PrimVar::psi] * V[PrimVar::psi]);
    U[ConsVar::magx] = V[PrimVar::magx];
    U[ConsVar::magy] = V[PrimVar::magy];
    U[ConsVar::magz] = V[PrimVar::magz];
    U[ConsVar::psi] = V[PrimVar::psi];
  }

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
  static Real ekin(const ConsVec U) {
    return 0.5 *
           (U[ConsVar::momx] * U[ConsVar::momx] + U[ConsVar::momy] * U[ConsVar::momy] +
            U[ConsVar::momz] * U[ConsVar::momz]) /
           U[ConsVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsb(const ConsVec U) {
    return 0.5 *
           (U[ConsVar::magx] * U[ConsVar::magx] + U[ConsVar::magy] * U[ConsVar::magy] +
            U[ConsVar::magz] * U[ConsVar::magz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const ConsVec U) {
    return (U[ConsVar::etot] - prsb(U) - ekin(U) -
            0.5 * U[ConsVar::psi] * U[ConsVar::psi]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsg(const Real gamma, const ConsVec U) { return (gamma - 1.0) * eint(U); }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const ConsVec U, const EosParameters &eosParams) {
    using Kokkos::pow;

    return prsg(eosParams.gamma, U) / pow(U[ConsVar::dens], eosParams.gamma);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const ConsVec U) {
    using Kokkos::fmin;

    for (int nV = 0; nV < NUM_VARS; nV++) {
      if (Kokkos::isnan(U[nV])) {
        return -1.0;
      }
    }
    return fmin(U[ConsVar::dens], eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const ConsVec U) {
    return Kokkos::sqrt(gamma * prsg(gamma, U) / U[ConsVar::dens]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonic(const Real gamma, const ConsVec U, const Real bl) {
    using Kokkos::sqrt;

    const Real asq = gamma * prsg(gamma, U) / U[ConsVar::dens];
    const Real bsq = Real(2.0) * prsb(U) / U[ConsVar::dens];
    const Real blsq = bl * bl / U[ConsVar::dens];
    const Real absq = asq + bsq;

    return sqrt(0.5 * (absq + sqrt(absq * absq - 4.0 * asq * blsq)));
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicX(const Real gamma, const ConsVec U) {
    return fastMagSonic(gamma, U, U[ConsVar::magx]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicY(const Real gamma, const ConsVec U) {
    return fastMagSonic(gamma, U, U[ConsVar::magy]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicZ(const Real gamma, const ConsVec U) {
    return fastMagSonic(gamma, U, U[ConsVar::magz]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
