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
  static void cons2prim(const Real gam, const MHD_GLM::ConsVec U, MHD_GLM::PrimVec V) {
    using MHD_GLM::PrimVar;
    using MHD_GLM::SimVar;

    V[PrimVar::dens] = U[SimVar::dens];
    V[PrimVar::velx] = U[SimVar::momx] / U[SimVar::dens];
    V[PrimVar::vely] = U[SimVar::momy] / U[SimVar::dens];
    V[PrimVar::velz] = U[SimVar::momz] / U[SimVar::dens];
    V[PrimVar::prsg] = prsg(gam, U);
    V[PrimVar::magx] = U[SimVar::magx];
    V[PrimVar::magy] = U[SimVar::magy];
    V[PrimVar::magz] = U[SimVar::magz];
    V[PrimVar::psi] = U[SimVar::psi];
  }

  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const MHD_GLM::PrimVec V, MHD_GLM::ConsVec U) {
    using MHD_GLM::PrimVar;
    using MHD_GLM::SimVar;

    U[SimVar::dens] = V[PrimVar::dens];
    U[SimVar::momx] = V[PrimVar::dens] * V[PrimVar::velx];
    U[SimVar::momy] = V[PrimVar::dens] * V[PrimVar::vely];
    U[SimVar::momz] = V[PrimVar::dens] * V[PrimVar::velz];
    U[SimVar::etot] =
        V[PrimVar::prsg] / (gam - Real(1.0)) +
        Real(0.5) *
            (V[PrimVar::dens] * (V[PrimVar::velx] * V[PrimVar::velx] +
                                 V[PrimVar::vely] * V[PrimVar::vely] +
                                 V[PrimVar::velz] * V[PrimVar::velz]) +
             V[PrimVar::magx] * V[PrimVar::magx] + V[PrimVar::magy] * V[PrimVar::magy] +
             V[PrimVar::magz] * V[PrimVar::magz] + V[PrimVar::psi] * V[PrimVar::psi]);
    U[SimVar::magx] = V[PrimVar::magx];
    U[SimVar::magy] = V[PrimVar::magy];
    U[SimVar::magz] = V[PrimVar::magz];
    U[SimVar::psi] = V[PrimVar::psi];
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirx(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return U[SimVar::momx] > 0.0 ? 1.0 : (U[SimVar::momx] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdiry(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return U[SimVar::momy] > 0.0 ? 1.0 : (U[SimVar::momy] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real flowdirz(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return U[SimVar::momz] > 0.0 ? 1.0 : (U[SimVar::momz] < 0.0 ? -1.0 : 0.0);
  }

  KOKKOS_INLINE_FUNCTION
  static Real ekin(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return Real(0.5) *
           (U[SimVar::momx] * U[SimVar::momx] + U[SimVar::momy] * U[SimVar::momy] +
            U[SimVar::momz] * U[SimVar::momz]) /
           U[SimVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsb(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return Real(0.5) *
           (U[SimVar::magx] * U[SimVar::magx] + U[SimVar::magy] * U[SimVar::magy] +
            U[SimVar::magz] * U[SimVar::magz]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return (U[SimVar::etot] - prsb(U) - ekin(U) -
            Real(0.5) * U[SimVar::psi] * U[SimVar::psi]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const MHD_GLM::ConsVec U, const EosParameters &eosParams) {
    using Kokkos::pow;
    using MHD_GLM::SimVar;

    return prsg(eosParams.gamma, U) / pow(U[SimVar::dens], eosParams.gamma);
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const MHD_GLM::ConsVec U) {
    using Kokkos::fmin;
    using MHD_GLM::SimVar;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      if (Kokkos::isnan(U[nV])) {
        return -1.0;
      }
    }
    return fmin(U[SimVar::dens], eint(U));
  }

  KOKKOS_INLINE_FUNCTION
  static Real prsg(const Real gamma, const MHD_GLM::ConsVec U) {
    return (gamma - Real(1.0)) * eint(U);
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gamma, const MHD_GLM::ConsVec U) {
    using MHD_GLM::SimVar;
    return Kokkos::sqrt(gamma * prsg(gamma, U) / U[SimVar::dens]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonic(const Real gamma, const MHD_GLM::ConsVec U, const Real bl) {
    using Kokkos::sqrt;
    using MHD_GLM::SimVar;

    const Real asq = gamma * prsg(gamma, U) / U[SimVar::dens];
    const Real bsq = Real(2.0) * prsb(U) / U[SimVar::dens];
    const Real absq = asq + bsq;
    Real cfsq = Real(0.5) *
                (absq + sqrt(absq * absq - Real(4.0) * asq * bl * bl / U[SimVar::dens]));

    return sqrt(cfsq);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicX(const Real gamma, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, U, U[MHD_GLM::SimVar::magx]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicY(const Real gamma, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, U, U[MHD_GLM::SimVar::magy]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real fastMagSonicZ(const Real gamma, const MHD_GLM::ConsVec U) {
    return fastMagSonic(gamma, U, U[MHD_GLM::SimVar::magz]);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
