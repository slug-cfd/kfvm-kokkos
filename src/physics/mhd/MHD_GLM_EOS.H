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
  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gamma, const Real J, const MHD_GLM::ConsVec U,
                        MHD_GLM::PrimVec V) {
    using MHD_GLM::PrimVar;
    using MHD_GLM::SimVar;

    V[PrimVar::dens] = U[SimVar::dens] / J;

    V[PrimVar::velx] = U[SimVar::momx] / U[SimVar::dens];
    V[PrimVar::vely] = U[SimVar::momy] / U[SimVar::dens];
    V[PrimVar::velz] = U[SimVar::momz] / U[SimVar::dens];

    V[PrimVar::prsg] = prsg(gamma, J, U);

    V[PrimVar::magx] = U[SimVar::magx] / J;
    V[PrimVar::magy] = U[SimVar::magy] / J;
    V[PrimVar::magz] = U[SimVar::magz] / J;

    V[PrimVar::psi] = U[SimVar::psi] / J;
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gamma, const Real J, const MHD_GLM::PrimVec V,
                        MHD_GLM::ConsVec U) {
    using MHD_GLM::PrimVar;
    using MHD_GLM::SimVar;

    U[SimVar::dens] = J * V[PrimVar::dens];

    U[SimVar::momx] = J * V[PrimVar::dens] * V[PrimVar::velx];
    U[SimVar::momy] = J * V[PrimVar::dens] * V[PrimVar::vely];
    U[SimVar::momz] = J * V[PrimVar::dens] * V[PrimVar::velz];

    U[SimVar::etot] =
        J *
        (V[PrimVar::prsg] / (gamma - 1.0) +
         0.5 *
             (V[PrimVar::dens] * (V[PrimVar::velx] * V[PrimVar::velx] +
                                  V[PrimVar::vely] * V[PrimVar::vely] +
                                  V[PrimVar::velz] * V[PrimVar::velz]) +
              V[PrimVar::magx] * V[PrimVar::magx] + V[PrimVar::magy] * V[PrimVar::magy] +
              V[PrimVar::magz] * V[PrimVar::magz] + V[PrimVar::psi] * V[PrimVar::psi]));

    U[SimVar::magx] = J * V[PrimVar::magx];
    U[SimVar::magy] = J * V[PrimVar::magy];
    U[SimVar::magz] = J * V[PrimVar::magz];

    U[SimVar::psi] = J * V[PrimVar::psi];
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const Real J, const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    const Real rho = U[SimVar::dens] / J;
    const Real vx = U[SimVar::momx] / U[SimVar::dens];
    const Real vy = U[SimVar::momy] / U[SimVar::dens];
    const Real vz = U[SimVar::momz] / U[SimVar::dens];
    return 0.5 * rho * (vx * vx + vy * vy + vz * vz);
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsb(const Real J, const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return (0.5 / (J * J)) *
           (U[SimVar::magx] * U[SimVar::magx] + U[SimVar::magy] * U[SimVar::magy] +
            U[SimVar::magz] * U[SimVar::magz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const Real J, const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return (U[SimVar::etot] / J - prsb(J, U) - ekin(J, U) -
            0.5 * U[SimVar::psi] * U[SimVar::psi] / (J * J));
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsg(const Real gamma, const Real J, const MHD_GLM::ConsVec U) {
    return (gamma - Real(1.0)) * eint(J, U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const EosParameters &eosParams, const Real J,
                   const MHD_GLM::ConsVec U) {
    return prsg(eosParams.gamma, J, U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const Real J, const MHD_GLM::ConsVec U) {
    using Kokkos::fmin;
    using MHD_GLM::SimVar;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      if (Kokkos::isnan(U[nV])) {
        return -1.0;
      }
    }
    return 1.0; // fmin(U[SimVar::dens],eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const Real J, const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return Kokkos::sqrt(gamma * J * prsg(gamma, J, U) / U[SimVar::dens]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonic(const Real gamma, const Real J, const MHD_GLM::ConsVec U,
                           const Real bl) {
    using Kokkos::sqrt;
    using MHD_GLM::SimVar;

    const Real asq = gamma * J * prsg(gamma, J, U) / U[SimVar::dens];
    const Real bsq = Real(2.0) * J * prsb(J, U) / U[SimVar::dens];
    const Real absq = asq + bsq;
    // One factor of J from each bl, and one from dens
    const Real abl = 4.0 * asq * bl * bl / (J * U[SimVar::dens]);
    Real cfsq = 0.5 * (absq + sqrt(absq * absq - abl));

    return sqrt(cfsq);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicX(const Real gamma, const Real J, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, J, U, U[MHD_GLM::SimVar::magx]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicY(const Real gamma, const Real J, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, J, U, U[MHD_GLM::SimVar::magy]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicZ(const Real gamma, const Real J, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, J, U, U[MHD_GLM::SimVar::magz]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
