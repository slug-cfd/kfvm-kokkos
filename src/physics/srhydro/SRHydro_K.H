#ifndef SRHYDRO_K_H_
#define SRHYDRO_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits>

#include "../../Dimension.H"
#include "../../Geometry.H"
#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "../SimVar.H"
#include "Definitions.H"
#include "SRHydro_EOS.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::SRHydro> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using PrimVar = typename SimVar<EquationType::SRHydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  Real D, Sx, Sy, Sz, etot;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const ConsVec ref_, Real gam_) {
    (void)gam_;

    D = ref_[ConsVar::dens];
    Sx = ref_[ConsVar::momx];
    Sy = ref_[ConsVar::momy];
    Sz = ref_[ConsVar::momz];
    etot = ref_[ConsVar::etot];
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {
    state(ConsVar::momx) = state(ConsVar::momx) / D - state(ConsVar::dens) * Sx / (D * D);
    state(ConsVar::momy) = state(ConsVar::momy) / D - state(ConsVar::dens) * Sy / (D * D);
    state(ConsVar::momz) = state(ConsVar::momz) / D - state(ConsVar::dens) * Sz / (D * D);
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {
    state(ConsVar::momx) = state(ConsVar::dens) * Sx / D + D * state(ConsVar::momx);
    state(ConsVar::momy) = state(ConsVar::dens) * Sy / D + D * state(ConsVar::momy);
    state(ConsVar::momz) = state(ConsVar::dens) * Sz / D + D * state(ConsVar::momz);
  }
};

template <class UViewType>
struct AuxVars<EquationType::SRHydro, UViewType> {
  using AuxVar = typename SimVar<EquationType::SRHydro>::AuxVar;
  using PrimVar = typename SimVar<EquationType::SRHydro>::PrimVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::SRHydro>::PrimVec;

  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Try not to assume the order of SimVars
    ConsVec u;
    PrimVec v;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      u[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Call EOS for full primitive state all at once
    EOS<EquationType::SRHydro>::cons2prim(eosParams.gamma, u, v);

    // Copy everything back out
    V(KFVM_D_DECL(i, j, k), AuxVar::rho) = v[PrimVar::rho];
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) = v[PrimVar::velx];
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) = v[PrimVar::vely];
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) = v[PrimVar::velz];
    V(KFVM_D_DECL(i, j, k), AuxVar::lorz) = v[PrimVar::lorz];
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) = v[PrimVar::pres];
  }
};

template <class UViewType>
struct FlowStats<EquationType::SRHydro, UViewType> {
  UViewType U;
  Geometry geom;
  EosParameters eosParams;

  FlowStats(const UViewType &U_, const Geometry &geom_, const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  FlowStatsArray &stat) const {}
};

template <>
struct SpeedEstimate_K<EquationType::SRHydro> {
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  EosParameters eosParams;

  SpeedEstimate_K(KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                              const FaceDataView &rsZ_),
                  const EosParameters &eosParams_)
      : KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &maxLam,
                  Real &maxVel) const {}
};

namespace SRHydro {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::SRHydro, UViewType> {
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;

  SourceDataView source;
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  UViewType U;
  WenoFlagView wenoFlag;
  DiffMatView D;
  QuadRuleView ab, wt;
  EosParameters eosParams;
  UserParameters userParams;
  Geometry geom;
  Real t;

  SourceTerms_K(const SourceDataView source_,
                KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                            const FaceDataView &rsZ_),
                const UViewType &U_, const WenoFlagView &wenoFlag_, const DiffMatView &D_,
                const QuadRuleView &ab_, const QuadRuleView &wt_,
                const EosParameters &eosParams_, const UserParameters &userParams_,
                const Geometry &geom_, const Real t_)
      : source(source_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), U(U_),
        wenoFlag(wenoFlag_), D(D_), ab(ab_), wt(wt_), eosParams(eosParams_),
        userParams(userParams_), geom(geom_), t(t_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Get coordinates of cell center
    Real xc = geom.xCoord(i);
    Real yc = geom.yCoord(j);
#if (SPACE_DIM == 2)
    Real zc = geom.zCoord(0);
#else
    Real zc = geom.zCoord(k);
#endif

    const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      Real x = Numeric::sclAb(xc, geom.dx, ab(nQ));
      for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
        Real y = Numeric::sclAb(yc, geom.dy, ab(nR));
        for (int nS = 0; nS < nzQuad; nS++) {
          Real z = SPACE_DIM == 2 ? zc : Numeric::sclAb(zc, geom.dz, ab(nS));
          ConsVec state, force;
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            force[nV] = 0;
          }

          // Fill force with user defined terms
          SRHydro::SourceTerm(t, x, y, z, eosParams, userParams, state, force);

          // overwrite source with forcing
          for (int nV = 0; nV < NUM_VARS; nV++) {
            source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) = force[nV];
          }
        }
      }
    }
  }
};

template <class UDataView>
struct PositivityPreserve_K<EquationType::SRHydro, UDataView> {
  using ConsVar = typename SimVar<EquationType::SRHydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::SRHydro>::ConsVec;

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
    Real rDens = refState[ConsVar::dens];
    Real edRef = eDiff(refState);

    // Find ranges of density and etot
    Real maxDens = -std::numeric_limits<Real>::max();
    Real minDens = std::numeric_limits<Real>::max();
    Real minEDiff = std::numeric_limits<Real>::max();
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
          minEDiff = fmin(minEDiff, eDiff(state));
#if (SPACE_DIM == 3)
        }
#endif
      }
    }

    // Enforce bounds on density
    Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;
    // Find mixing amount
    Real theta = Real(1.0);
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // West face
      Real rho = rsX(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // East face
      rho = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // South face
      rho = rsY(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // North face
      rho = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
#if (SPACE_DIM == 3)
      // Bottom face
      rho = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // Top face
      rho = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, ConsVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
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
            theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
            theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
          }
        }
      }
    }

    // Apply mixing
    wenoFlag(KFVM_D_DECL(i, j, k), 1) = theta;
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

    // Enforce lower bound on eDiff
    Real eDiffDiff = edRef - minEDiff;
    // Find mixing amount
    theta = Real(1.0);
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      ConsVec state;
      Real ed;
      // West face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      ed = eDiff(state);
      theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
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
            Real ed = eDiff(state);
            theta = ed < minEDiff ? fmin(theta, eDiffDiff / (edRef - ed)) : theta;
          }
        }
      }
    }

    // Apply mixing
    wenoFlag(KFVM_D_DECL(i, j, k), 1) = fmin(theta, wenoFlag(KFVM_D_DECL(i, j, k), 1));
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
  Real eDiff(const ConsVec state) const {
    using Kokkos::sqrt;

    const Real sSq = (state[ConsVar::momx] * state[ConsVar::momx] +
                      state[ConsVar::momy] * state[ConsVar::momy] +
                      state[ConsVar::momz] * state[ConsVar::momz]);

    return state[ConsVar::etot] - sqrt(state[ConsVar::dens] * state[ConsVar::dens] + sSq);
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
