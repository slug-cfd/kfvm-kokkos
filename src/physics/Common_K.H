#ifndef COMMON_K_H_
#define COMMON_K_H_

#include <cstdint>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Dimension.H"
#include "../Geometry.H"
#include "../ParameterStructs.H"
#include "../Types.H"
#include "EquationTypes.H"
#include "SimVar.H"

namespace KFVM {

namespace Physics {

// Set LLF and HLL as default, only difference is in the speed calculation
template <EquationType ET, RSType RST>
struct RiemannSolver {
  using ConsVar = typename SimVar<ET>::ConsVar;
  using ConsVec = typename SimVar<ET>::ConsVec;
  using PrimVec = typename SimVar<ET>::PrimVec;

  KOKKOS_INLINE_FUNCTION
  static Real solve(const EosParameters &eosParams, const int dir, const ConsVec uL,
                    const ConsVec uR, const PrimVec vL, const PrimVec vR, ConsVec F) {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;

    // set field orderings to collapse to 1d solver
    const int ord[] = {dir, (dir + 1) % 3, (dir + 2) % 3};

    // Wave speed estimates
    Real sL, sR, sMax;
    sMax = RSPhys<ET>::speeds(eosParams, ord, uL, uR, vL, vR, sL, sR);
    if constexpr (RST == RSType::LLF) {
      sL = -sMax;
      sR = sMax;
    }

    // Combine as in Batten to avoid conditionals
    const Real tR = (fmin(0.0, sR) - fmin(0.0, sL)) / (sR - sL), tL = 1.0 - tR;
    const Real tD = 0.5 * (sR * fabs(sL) - sL * fabs(sR)) / (sR - sL);

    // L/R fluxes and add together
    ConsVec FR;
    RSPhys<ET>::flux(eosParams, ord, uL, vL, F);
    RSPhys<ET>::flux(eosParams, ord, uR, vR, FR);

    for (int nV = 0; nV < NUM_VARS; nV++) {
      F[nV] = tL * F[nV] + tR * FR[nV] - tD * (uR[nV] - uL[nV]);
    }

    return sMax;
  }
};

// Application of a Riemann solver always looks the same
template <RSType RST>
struct RiemannSolverX_K {
  using ConsVec = typename SimVar<eqType>::ConsVec;
  using PrimVec = typename SimVar<eqType>::PrimVec;

  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverX_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<eqType>::cons2prim(eosParams.gamma, uL, vL);
      EOS<eqType>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = RiemannSolver<eqType, RST>::solve(eosParams, 0, uL, uR, vL, vR, F);
      maxVel = vq > maxVel ? vq : maxVel;

      // Overwrite states with flux
      for (int nV = 0; nV < NUM_VARS; nV++) {
        U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
        U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
      }
    }
  }
};

template <RSType RST>
struct RiemannSolverY_K {
  using ConsVec = typename SimVar<eqType>::ConsVec;
  using PrimVec = typename SimVar<eqType>::PrimVec;

  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverY_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<eqType>::cons2prim(eosParams.gamma, uL, vL);
      EOS<eqType>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = RiemannSolver<eqType, RST>::solve(eosParams, 1, uL, uR, vL, vR, F);
      maxVel = vq > maxVel ? vq : maxVel;

      // Overwrite states with flux
      for (int nV = 0; nV < NUM_VARS; nV++) {
        U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
        U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
      }
    }
  }
};

template <RSType RST>
struct RiemannSolverZ_K {
  using ConsVec = typename SimVar<eqType>::ConsVec;
  using PrimVec = typename SimVar<eqType>::PrimVec;

  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverZ_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

#if (SPACE_DIM == 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k, Real &maxVel) const {
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<eqType>::cons2prim(eosParams.gamma, uL, vL);
      EOS<eqType>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = RiemannSolver<eqType, RST>::solve(eosParams, 2, uL, uR, vL, vR, F);
      maxVel = vq > maxVel ? vq : maxVel;

      // Overwrite states with flux
      for (int nV = 0; nV < NUM_VARS; nV++) {
        U(i, j, k, 0, nQ, nV) = F[nV];
        U(i, j, k, 1, nQ, nV) = F[nV];
      }
    }
  }
#endif
};

// Default positivity preservation for equations with
// density and internal energy constraints
// -> Hydro, NavierStokes, MHD_GLM, MHD_8W
template <EquationType ET, class UDataView>
struct PositivityPreserve_K {
  using ConsVar = typename SimVar<ET>::ConsVar;
  using ConsVec = typename SimVar<ET>::ConsVec;

  UDataView U;
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  bool haveSources;
  SourceDataView source;
  WenoFlagView wenoFlag;
  EosParameters eosParams;

  PositivityPreserve_K(const UDataView &U_,
                       KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                                   const FaceDataView &rsZ_),
                       const bool haveSources_, const SourceDataView &source_,
                       const WenoFlagView &wenoFlag_, const EosParameters &eosParams_)
      : U(U_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), haveSources(haveSources_),
        source(source_), wenoFlag(wenoFlag_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using Kokkos::fmax;
    using Kokkos::fmin;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    // Save reference state
    ConsVec refState;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }
    const Real rDens = refState[ConsVar::dens];

    // Find ranges of density, pressure, and soundspeed
    Real maxDens = -std::numeric_limits<Real>::max();
    Real minDens = std::numeric_limits<Real>::max();
    Real minEint = std::numeric_limits<Real>::max();
    Real minSdSp = std::numeric_limits<Real>::max();
    for (idx_t xo = -1; xo <= 1; xo++) {
      for (idx_t yo = -1; yo <= 1; yo++) {
#if (SPACE_DIM == 3)
        for (idx_t zo = -1; zo <= 1; zo++) {
#endif
          ConsVec state;
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = U(KFVM_D_DECL(i + xo, j + yo, k + zo), nV);
          }
          maxDens = fmax(maxDens, state[ConsVar::dens]);
          minDens = fmin(minDens, state[ConsVar::dens]);
          const Real eint = EOS<ET>::eint(state);
          const Real sdsp = EOS<ET>::soundspeed(eosParams.gamma, state);
          minEint = fmin(minEint, eint);
          minSdSp = fmin(minSdSp, sdsp);
#if (SPACE_DIM == 3)
        }
#endif
      }
    }

    // Find size of flattener
    const Real dxU = U(KFVM_D_DECL(i + 1, j, k), ConsVar::momx) /
                         U(KFVM_D_DECL(i + 1, j, k), ConsVar::dens) -
                     U(KFVM_D_DECL(i - 1, j, k), ConsVar::momx) /
                         U(KFVM_D_DECL(i - 1, j, k), ConsVar::dens);

    const Real dyV = U(KFVM_D_DECL(i, j + 1, k), ConsVar::momy) /
                         U(KFVM_D_DECL(i, j + 1, k), ConsVar::dens) -
                     U(KFVM_D_DECL(i, j - 1, k), ConsVar::momy) /
                         U(KFVM_D_DECL(i, j - 1, k), ConsVar::dens);

    const Real dzW =
        U(KFVM_D_DECL(i, j, k + 1), ConsVar::momz) /
            U(KFVM_D_DECL(i, j, k + 1), ConsVar::dens) -
        U(KFVM_D_DECL(i, j, k - 1), ConsVar::momz) /
            U(KFVM_D_DECL(i, j, k - 1), ConsVar::dens); // Naturally zero in 2d

    const Real divV = (dxU + dyV + dzW) / 2.0; // Undivided difference
    Real kappa = 0.2;
    const Real eta = fmin(1.0, fmax(0.0, -(kappa * minSdSp + divV) / (kappa * minSdSp)));
    kappa *= (1.0 - eta);

    // Extend density and pressure ranges
    maxDens *= (1.0 + kappa);
    minDens *= (1.0 - kappa);
    minEint *= (1.0 - kappa);

    // Enforce positive density
    Real thD = densTheta(KFVM_D_DECL(i, j, k), rDens, maxDens, minDens, nQuad);
    if (thD > 0.0 && thD < 1.0) {
      mixState(KFVM_D_DECL(i, j, k), refState, thD, nQuad);
    } else if (thD < 1.0) { // thD == 0
      replaceState(KFVM_D_DECL(i, j, k), refState, nQuad);
    }

    // Enforce positive internal energy
    const Real thP = eintTheta(KFVM_D_DECL(i, j, k), refState, minEint, nQuad);
    if (thP > 0.0 && thP < 1.0) {
      mixState(KFVM_D_DECL(i, j, k), refState, thP, nQuad);
    } else if (thP < 1.0) { // thP == 0
      replaceState(KFVM_D_DECL(i, j, k), refState, nQuad);
    }

    // Update flag to most restrictive case
    wenoFlag(KFVM_D_DECL(i, j, k), 1) = fmin(thD, thP);
  }

  KOKKOS_INLINE_FUNCTION
  Real densTheta(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                 const Real rDens, const Real maxDens, const Real minDens,
                 const idx_t nQuad) const {
    using Kokkos::fmin;

    const Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;
    Real theta = 1.0;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      Real rho;
      // West face
      rho = rsX(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  East face
      rho = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  South face
      rho = rsY(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  North face
      rho = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
#if (SPACE_DIM == 3)
      // Bottom face
      rho = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      // Top face
      rho = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, ConsVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
#endif
    }

    // Find mixing amount given by interior values
    if (haveSources) {
      int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            Real rho =
                source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::dens);
            theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
            theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
          }
        }
      }
    }

    return theta;
  }

  KOKKOS_INLINE_FUNCTION
  Real eintTheta(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                 const ConsVec refState, const Real minEint, const idx_t nQuad) const {
    using Kokkos::fmin;

    Real theta = 1.0;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      ConsVec state;
      Real ei;

      // West face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));

      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));

      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));

      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));

      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      ei = EOS<ET>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));
#endif
    }

    // Find mixing amount given by interior values
    if (haveSources) {
      int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            ConsVec state;
            for (int nV = 0; nV < NUM_VARS; nV++) {
              state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            }
            Real ei = EOS<ET>::eint(state);
            theta = ei > minEint ? theta : fmin(theta, eintPos(minEint, refState, state));
          }
        }
      }
    }

    return theta;
  }

  KOKKOS_INLINE_FUNCTION
  void mixState(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                const ConsVec refState, const Real theta, const idx_t nQuad) const {
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV) =
            refState[nV] + theta * (rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV) - refState[nV]);
        rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV) =
            refState[nV] +
            theta * (rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV) - refState[nV]);
        rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV) =
            refState[nV] + theta * (rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV) - refState[nV]);
        rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV) =
            refState[nV] +
            theta * (rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV) - refState[nV]);
#if (SPACE_DIM == 3)
        rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV) =
            refState[nV] + theta * (rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV) - refState[nV]);
        rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV) =
            refState[nV] +
            theta * (rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV) - refState[nV]);
#endif
      }
    }
    if (haveSources) {
      int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            for (idx_t nV = 0; nV < NUM_VARS; nV++) {
              source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) =
                  refState[nV] +
                  theta * (source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) -
                           refState[nV]);
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void replaceState(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                    const ConsVec refState, const idx_t nQuad) const {
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV) = refState[nV];
        rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV) = refState[nV];
        rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV) = refState[nV];
        rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV) = refState[nV];
#if (SPACE_DIM == 3)
        rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV) = refState[nV];
        rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV) = refState[nV];
#endif
      }
    }
    if (haveSources) {
      int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            for (idx_t nV = 0; nV < NUM_VARS; nV++) {
              source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) = refState[nV];
            }
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  Real eintPos(const Real emin, const ConsVec refState, const ConsVec state) const {
    using Kokkos::fabs;
    using Kokkos::fmin;

    Real a = 0.0, b = 1.0, c;
    ConsVec stTheta;

    // Bisect on theta for 10 iterations regardless
    for (int nIter = 0; nIter < 10; nIter++) {
      // Midpoint and midpoint fval
      c = 0.5 * (a + b);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        stTheta[nV] = refState[nV] + c * (state[nV] - refState[nV]);
      }

      // eint difference at c-state
      const Real dec = EOS<ET>::eint(stTheta) - emin;

      // Walk forward if c-state is valid, fall back otherwise
      if (dec > 0.0) {
        a = c;
      } else {
        b = c;
      }
    }

    // Round down to lower bound to guarantee a valid state
    return a;
  }
};

template <EquationType ET, class UViewType>
struct RandForcingIntegrate_K {
  using ConsVar = typename SimVar<ET>::ConsVar;

  UViewType U;
  EosParameters eosParams;
  Geometry geom;

  RandForcingIntegrate_K(const UViewType &U_, const EosParameters &eosParams_,
                         const Geometry geom_)
      : U(U_), eosParams(eosParams_), geom(geom_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &intA,
                  Real &intB) const {
    using Kokkos::cos;
    using Kokkos::sin;
#ifdef ENABLE_RANDOM_FORCING

    const Real str = 2.0; // Extra 2 to enforce conj. symm.
    const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

    // Sum forcing modes and track contributions
    Real fu = 0.0, fv = 0.0, fw = 0.0;
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
      const Real x = geom.xCoord(i), y = geom.yCoord(j);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0, z = geom.zCoord(0);
#else
      const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
#endif

      // Set k.x and set forcing from this mode ignoring projection
      const Real kdx = k1 * x + k2 * y + k3 * z;
      const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 2)
      const Real fwn = 0.0;
#else
      const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif

      // Project and add into total forcing
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

      fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
                   mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

      fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
                   mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

      fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
                   mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
    }

    intA += 0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw * fw);
    intB += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
            U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
            U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;
#endif
  }
};

template <EquationType ET, class UViewType>
struct RandForcingApply_K {
  using ConsVar = typename SimVar<ET>::ConsVar;

  UViewType U;
  EosParameters eosParams;
  Geometry geom;

  RandForcingApply_K(const UViewType &U_, const EosParameters &eosParams_,
                     const Geometry &geom_)
      : U(U_), eosParams(eosParams_), geom(geom_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using Kokkos::cos;
    using Kokkos::sin;
#ifdef ENABLE_RANDOM_FORCING

    const Real str = 2.0 * eosParams.forceStr; // Extra 2 to enforce conj. symm.
    const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

    // Sum forcing modes and track contributions
    Real fu = 0.0, fv = 0.0, fw = 0.0;
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
      const Real x = geom.xCoord(i), y = geom.yCoord(j);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0, z = geom.zCoord(0);
#else
      const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
#endif

      // Set k.x and set forcing from this mode ignoring projection
      const Real kdx = k1 * x + k2 * y + k3 * z;
      const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 2)
      const Real fwn = 0.0;
#else
      const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif

      // Project and add into total forcing
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

      fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
                   mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

      fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
                   mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

      fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
                   mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
    }

    // Energy in the perturbation as from above
    Real ePert =
        0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw * fw);
    ePert += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
             U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
             U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;

    // Update momenta and total energy
    U(KFVM_D_DECL(i, j, k), ConsVar::momx) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fu;
    U(KFVM_D_DECL(i, j, k), ConsVar::momy) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fv;
    U(KFVM_D_DECL(i, j, k), ConsVar::momz) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fw;
    U(KFVM_D_DECL(i, j, k), ConsVar::etot) += ePert;
#endif
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
