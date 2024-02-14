#ifndef SRHYDRO_EOS_H_
#define SRHYDRO_EOS_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>

#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../SimVar.H"

#define KFVM_MAX_SR_ITER 200

namespace KFVM {

namespace Physics {

template <>
struct EOS<EquationType::SRHydro> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::SRHydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  // Construction of primitives is not trivial
  // -> Elevate this into EOS to ensure that the
  //    expensive part is only done once
  // Maximum iterations explicitly passed to allow
  // some calling methods to use loose approximations
  // when appropriate
  KOKKOS_INLINE_FUNCTION
  static void cons2prim(const Real gam, const ConsVec U, PrimVec V) {
    // Single expensive solve can be done for pressure,
    // do it first and pass result to other EOS routines
    const Real p = pres(gam, U);

    // Get remaining quantities cheaply
    const Real W = lorenz(p, U);
    const Real rho = U[ConsVar::dens] / W, ei = eint(gam, p, U), h = Real(1.0) + gam * ei;

    // Get velocities and fill everything in
    V[PrimVar::rho] = rho;
    V[PrimVar::velx] = U[ConsVar::momx] / (rho * W * W * h);
    V[PrimVar::vely] = U[ConsVar::momy] / (rho * W * W * h);
    V[PrimVar::velz] = U[ConsVar::momz] / (rho * W * W * h);
    V[PrimVar::lorz] = W;
    V[PrimVar::pres] = p;
  }

  // Conservative from primitive is easy
  KOKKOS_INLINE_FUNCTION
  static void prim2cons(const Real gam, const PrimVec V, ConsVec U) {
    // Get enthalpy
    const Real h = enthalpy(gam, V);
    const Real W = V[PrimVar::lorz];

    U[ConsVar::dens] = V[PrimVar::rho] * W;
    U[ConsVar::momx] = U[ConsVar::dens] * W * h * V[PrimVar::velx];
    U[ConsVar::momy] = U[ConsVar::dens] * W * h * V[PrimVar::vely];
    U[ConsVar::momz] = U[ConsVar::dens] * W * h * V[PrimVar::velz];
    U[ConsVar::etot] = U[ConsVar::dens] * W * h - V[PrimVar::pres];
  }

  // All pressure recovery methods come from https://doi.org/10.48550/arXiv.2305.14805
  // Initial guess given by eqs (2.21) and (2.16)
  KOKKOS_INLINE_FUNCTION
  static Real presGuess(const Real gam, const ConsVec U) {
    using Kokkos::sqrt;

    const Real ESsq =
        U[ConsVar::etot] * U[ConsVar::etot] -
        (U[ConsVar::momx] * U[ConsVar::momx] + U[ConsVar::momy] * U[ConsVar::momy] +
         U[ConsVar::momz] * U[ConsVar::momz]);

    if (U[ConsVar::dens] >= ESsq / U[ConsVar::etot]) {
      return 0.0;
    }

    const Real gm2E = (gam - Real(2.0)) * U[ConsVar::etot];
    const Real gm = gam - Real(1.0);
    return (gm2E +
            sqrt(gm2E * gm2E - Real(4.0) * gm * (U[ConsVar::dens] * sqrt(ESsq) - ESsq))) /
           Real(2.0);
  }

  // More stable scheme, algo (2.3)
  KOKKOS_INLINE_FUNCTION
  static Real presStable(const Real gam, const Real rtol, const ConsVec U, Real p) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::sqrt;

    // Useful quantities
    const Real gm = gam - Real(1.0);
    const Real gm2 = gam - Real(2.0);
    const Real D = U[ConsVar::dens];
    const Real E = U[ConsVar::etot];
    const Real Ssq =
        (U[ConsVar::momx] * U[ConsVar::momx] + U[ConsVar::momy] * U[ConsVar::momy] +
         U[ConsVar::momz] * U[ConsVar::momz]);
    for (int n = 0; n < KFVM_MAX_SR_ITER; n++) {
      // Recurring quantities
      const Real EP = E + p;
      const Real rt = sqrt(EP * EP - Ssq);

      // Evaluate psi and derivative
      const Real psiP = Ssq + EP * (p / gm - E) + D * rt;
      const Real dpsiP = (Real(2.0) * p - gm2 * E) / gm + D * EP / rt;

      // Update p and test for convergence
      const Real dp = -psiP / dpsiP;
      if (fabs(dp) / p < rtol) {
        break;
      }
      p += dp;
    }

    return p;
  }

  KOKKOS_INLINE_FUNCTION
  static Real pres(const Real gam, const ConsVec U) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::sqrt;

    // Set tolerances
    const Real eps1 = Real(0.01), eps2 = Real(1.e-4);
    const Real rtol = Real(1.e-9);

    // Use bracket to get decent initial guess
    Real p = presGuess(gam, U);

    // Check which scheme to apply
    const Real Dsq = U[ConsVar::dens] * U[ConsVar::dens];
    const Real Esq = U[ConsVar::etot] * U[ConsVar::etot];
    const Real Ssq =
        (U[ConsVar::momx] * U[ConsVar::momx] + U[ConsVar::momy] * U[ConsVar::momy] +
         U[ConsVar::momz] * U[ConsVar::momz]);
    const Real ESsq = Esq - Ssq;
    // Immediately swap to stable version
    if (Dsq / ESsq < eps2) {
      return presStable(gam, rtol, U, p);
    }

    // Do fast version unless lorenz factor is bad
    // Algorithm (2.1)
    const Real gm = gam - Real(1.0);
    const Real gm2 = gam - Real(2.0);
    const Real c0 = -ESsq * (Dsq - ESsq) * gm * gm;
    const Real c1 = Real(2.0) * U[ConsVar::etot] * (gm2 * ESsq * gm - Dsq * gm * gm);
    const Real c2 =
        Esq * (gam * gam - Real(6.0) * gm) + Real(2.0) * Ssq * gm - Dsq * gm * gm;
    const Real c3 = -Real(2.0) * U[ConsVar::etot] * gm2;
    for (int n = 0; n < KFVM_MAX_SR_ITER; n++) {
      // Test lorenz factor
      // revert to stable if needed
      const Real W = lorenz(p, U);
      if (W < Real(1.0) + eps1) {
        return presStable(gam, rtol, U, p);
      }

      // Evaluate pressure polynomial and derivative
      const Real phiP = c0 + p * (c1 + p * (c2 + p * (c3 + p)));
      const Real dphiP = c1 + p * (Real(2.0) * c2 + p * (Real(3.0) * c3 + p));

      // Update p and test for convergence
      const Real dp = -phiP / dphiP;
      if (fabs(dp) / p < rtol) {
        break;
      }
      p += dp;
    }

    return p;
  }

  KOKKOS_INLINE_FUNCTION
  static Real lorenz(const Real p, const ConsVec U) {
    using Kokkos::sqrt;

    const Real DhW = U[ConsVar::etot] + p;
    const Real DhWsq = DhW * DhW;
    const Real msq = U[ConsVar::momx] * U[ConsVar::momx] +
                     U[ConsVar::momy] * U[ConsVar::momy] +
                     U[ConsVar::momz] * U[ConsVar::momz];
    return Real(1.0) / sqrt(Real(1.0) - msq / DhWsq);
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
  static Real eint(const Real gam, const Real p, const ConsVec U) {
    const Real rho = U[ConsVar::dens] / lorenz(p, U);
    return p / ((gam - Real(1.0)) * rho);
  }

  KOKKOS_INLINE_FUNCTION
  static Real eint(const Real gam, const PrimVec V) {
    return V[PrimVar::pres] / ((gam - Real(1.0)) * V[PrimVar::rho]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real enthalpy(const Real gam, const Real p, const ConsVec U) {
    const Real rho = U[ConsVar::dens] / lorenz(p, U);
    return Real(1.0) + gam * p / ((gam - Real(1.0)) * rho);
  }

  KOKKOS_INLINE_FUNCTION
  static Real enthalpy(const Real gam, const PrimVec V) {
    return Real(1.0) + gam * V[PrimVar::pres] / ((gam - Real(1.0)) * V[PrimVar::rho]);
  }

  KOKKOS_INLINE_FUNCTION
  static Real weno(const ConsVec U, const EosParameters &eosParams) {
    return U[ConsVar::dens];
  }

  KOKKOS_INLINE_FUNCTION
  static Real soundspeed(const Real gam, const PrimVec V) {
    using Kokkos::sqrt;

    const Real h = enthalpy(gam, V);

    return sqrt(gam * V[PrimVar::pres] / (V[PrimVar::rho] * h));
  }

  KOKKOS_INLINE_FUNCTION
  static Real positivity(const ConsVec U) {
    using Kokkos::fmin;

    // Calculate lower bound on etot
    const Real etotDiff =
        U[ConsVar::etot] * U[ConsVar::etot] -
        (U[ConsVar::dens] * U[ConsVar::dens] + U[ConsVar::momx] * U[ConsVar::momx] +
         U[ConsVar::momy] * U[ConsVar::momy] + U[ConsVar::momz] * U[ConsVar::momz]);

    // Note that the bound on etot also guarantees positive pressure
    // and subluminal velocities
    return fmin(U[ConsVar::dens], etotDiff);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
