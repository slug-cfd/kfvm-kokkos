#ifndef HYDRO_K_H_
#define HYDRO_K_H_

#include <cstdint>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../../Dimension.H"
#include "../../Geometry.H"
#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "../SimVar.H"
#include "Hydro_EOS.H"
#include "Hydro_Riemann.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::Hydro> {
  const Real gamm;
  const Real rho, u, v, w, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const Hydro::ConsVec ref_, Real gam_)
      : gamm(gam_ - 1.0), rho(ref_[Hydro::SimVar::dens]),
        u(ref_[Hydro::SimVar::momx] / rho), v(ref_[Hydro::SimVar::momy] / rho),
        w(ref_[Hydro::SimVar::momz] / rho), vsq(0.5 * (u * u + v * v + w * w)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {
    using Hydro::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) =
        gamm * (vsq * state(SimVar::dens) - u * state(SimVar::momx) -
                v * state(SimVar::momy) - w * state(SimVar::momz) + state(SimVar::etot));

    state(SimVar::momx) = (state(SimVar::momx) - u * state(SimVar::dens)) / rho;
    state(SimVar::momy) = (state(SimVar::momy) - v * state(SimVar::dens)) / rho;
    state(SimVar::momz) = (state(SimVar::momz) - w * state(SimVar::dens)) / rho;
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {
    using Hydro::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) = (vsq * state(SimVar::dens) + rho * u * state(SimVar::momx) +
                           rho * v * state(SimVar::momy) + rho * w * state(SimVar::momz) +
                           state(SimVar::etot) / gamm);

    state(SimVar::momx) = (rho * state(SimVar::momx) + u * state(SimVar::dens));
    state(SimVar::momy) = (rho * state(SimVar::momy) + v * state(SimVar::dens));
    state(SimVar::momz) = (rho * state(SimVar::momz) + w * state(SimVar::dens));
  }
};

template <class UViewType>
struct AuxVars<EquationType::Hydro, UViewType> {
  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using Hydro::AuxVar;
    using Hydro::ConsVec;
    using Hydro::SimVar;

    // Try not to assume the order of SimVars
    ConsVec ua;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      ua[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Set velocities
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) =
        U(KFVM_D_DECL(i, j, k), SimVar::momx) / U(KFVM_D_DECL(i, j, k), SimVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) =
        U(KFVM_D_DECL(i, j, k), SimVar::momy) / U(KFVM_D_DECL(i, j, k), SimVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) =
        U(KFVM_D_DECL(i, j, k), SimVar::momz) / U(KFVM_D_DECL(i, j, k), SimVar::dens);

    // Internal energy and pressure
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::Hydro>::eint(ua);
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) =
        EOS<EquationType::Hydro>::pres(eosParams.gamma, ua);
  }
};

template <class UViewType>
struct FlowStats<EquationType::Hydro, UViewType> {
  UViewType U;
  Geometry geom;
  EosParameters eosParams;

  FlowStats(const UViewType &U_, const Geometry &geom_, const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  FlowStatsArray &stat) const {
    using Hydro::SimVar;
    using Hydro::StatVar;

    // Compute kinetic energy
    const Real ek =
        0.5 *
        (U(KFVM_D_DECL(i, j, k), SimVar::momx) * U(KFVM_D_DECL(i, j, k), SimVar::momx) +
         U(KFVM_D_DECL(i, j, k), SimVar::momy) * U(KFVM_D_DECL(i, j, k), SimVar::momy) +
         U(KFVM_D_DECL(i, j, k), SimVar::momz) * U(KFVM_D_DECL(i, j, k), SimVar::momz)) /
        U(KFVM_D_DECL(i, j, k), SimVar::dens);

    // Compute velocity gradients
    const Real dyu = 0.5 *
                     (U(KFVM_D_DECL(i, j + 1, k), SimVar::momx) /
                          U(KFVM_D_DECL(i, j + 1, k), SimVar::dens) -
                      U(KFVM_D_DECL(i, j - 1, k), SimVar::momx) /
                          U(KFVM_D_DECL(i, j - 1, k), SimVar::dens)) /
                     geom.dy;
    const Real dxv = 0.5 *
                     (U(KFVM_D_DECL(i + 1, j, k), SimVar::momy) /
                          U(KFVM_D_DECL(i + 1, j, k), SimVar::dens) -
                      U(KFVM_D_DECL(i - 1, j, k), SimVar::momy) /
                          U(KFVM_D_DECL(i - 1, j, k), SimVar::dens)) /
                     geom.dx;
#if (SPACE_DIM == 2)
    const Real dzu = 0.0, dzv = 0.0, dxw = 0.0, dyw = 0.0;
#else
    const Real dzu = 0.5 *
                     (U(i, j, k + 1, SimVar::momx) / U(i, j, k + 1, SimVar::dens) -
                      U(i, j, k - 1, SimVar::momx) / U(i, j, k - 1, SimVar::dens)) /
                     geom.dz;

    const Real dzv = 0.5 *
                     (U(i, j, k + 1, SimVar::momy) / U(i, j, k + 1, SimVar::dens) -
                      U(i, j, k - 1, SimVar::momy) / U(i, j, k - 1, SimVar::dens)) /
                     geom.dz;

    const Real dxw = 0.5 *
                     (U(i + 1, j, k, SimVar::momz) / U(i + 1, j, k, SimVar::dens) -
                      U(i - 1, j, k, SimVar::momz) / U(i - 1, j, k, SimVar::dens)) /
                     geom.dx;

    const Real dyw = 0.5 *
                     (U(i, j + 1, k, SimVar::momz) / U(i, j + 1, k, SimVar::dens) -
                      U(i, j - 1, k, SimVar::momz) / U(i, j - 1, k, SimVar::dens)) /
                     geom.dy;
#endif

    // compute vorticity and enstrophy
    const Real omx = dyw - dzv, omy = dzu - dxw, omz = dxv - dyu;
    const Real et =
        0.5 * U(KFVM_D_DECL(i, j, k), SimVar::dens) * (omx * omx + omy * omy + omz * omz);

    // Add into array for reduction
    const Real ifac = geom.dvol / geom.volume;
    stat.data[StatVar::ekin] += ifac * ek;
    stat.data[StatVar::enst] += ifac * et;
  }
};

template <RSType rst>
struct RiemannSolverX_K<EquationType::Hydro, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverX_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = HydroRS<rst>::RiemannSolve_X(eosParams.gamma, uL, uR, vL, vR, F);
      maxVel = vq > maxVel ? vq : maxVel;

      // Overwrite states with flux
      for (int nV = 0; nV < NUM_VARS; nV++) {
        U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
        U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
      }
    }
  }
};

template <RSType rst>
struct RiemannSolverY_K<EquationType::Hydro, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverY_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = HydroRS<rst>::RiemannSolve_Y(eosParams.gamma, uL, uR, vL, vR, F);
      maxVel = vq > maxVel ? vq : maxVel;

      // Overwrite states with flux
      for (int nV = 0; nV < NUM_VARS; nV++) {
        U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
        U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
      }
    }
  }
};

template <RSType rst>
struct RiemannSolverZ_K<EquationType::Hydro, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverZ_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

#if (SPACE_DIM == 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k, Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = HydroRS<rst>::RiemannSolve_Z(eosParams.gamma, uL, uR, vL, vR, F);
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

template <>
struct SpeedEstimate_K<EquationType::Hydro> {
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  EosParameters eosParams;

  SpeedEstimate_K(KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                              const FaceDataView &rsZ_),
                  const EosParameters &eosParams_)
      : KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {}
};

namespace Hydro {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::Hydro, UViewType> {
  SourceDataView source;
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  UViewType U;
  WenoFlagView wenoFlag;
  DiffMatView D;
  QuadRuleView ab;
  EosParameters eosParams;
  Geometry geom;
  Real t;

  SourceTerms_K(const SourceDataView &source_,
                KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                            const FaceDataView &rsZ_),
                const UViewType &U_, const WenoFlagView &wenoFlag_, const DiffMatView &D_,
                const QuadRuleView &ab_, const EosParameters &eosParams_,
                const Geometry &geom_, const Real t_)
      : source(source_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), U(U_),
        wenoFlag(wenoFlag_), D(D_), ab(ab_), eosParams(eosParams_), geom(geom_), t(t_) {}

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
          Hydro::ConsVec state, force;
          // Real state[10],force[10];
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            force[nV] = 0;
          }

          // Fill force with user defined terms
          Hydro::SourceTerm(t, x, y, z, state, force);

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
struct PositivityPreserve_K<EquationType::Hydro, UDataView> {
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
    using Hydro::ConsVec;
    using Hydro::SimVar;
    using Kokkos::fmax;
    using Kokkos::fmin;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    // Save reference state
    ConsVec refState;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }
    const Real rDens = refState[SimVar::dens];

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
          maxDens = fmax(maxDens, state[SimVar::dens]);
          minDens = fmin(minDens, state[SimVar::dens]);
          const Real eint = EOS<EquationType::Hydro>::eint(state);
          const Real sdsp = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, state);
          minEint = fmin(minEint, eint);
          minSdSp = fmin(minSdSp, sdsp);
#if (SPACE_DIM == 3)
        }
#endif
      }
    }

    // Find size of flattener
    const Real dxU = U(KFVM_D_DECL(i + 1, j, k), SimVar::momx) /
                         U(KFVM_D_DECL(i + 1, j, k), SimVar::dens) -
                     U(KFVM_D_DECL(i - 1, j, k), SimVar::momx) /
                         U(KFVM_D_DECL(i - 1, j, k), SimVar::dens);

    const Real dyV = U(KFVM_D_DECL(i, j + 1, k), SimVar::momy) /
                         U(KFVM_D_DECL(i, j + 1, k), SimVar::dens) -
                     U(KFVM_D_DECL(i, j - 1, k), SimVar::momy) /
                         U(KFVM_D_DECL(i, j - 1, k), SimVar::dens);

    const Real dzW =
        U(KFVM_D_DECL(i, j, k + 1), SimVar::momz) /
            U(KFVM_D_DECL(i, j, k + 1), SimVar::dens) -
        U(KFVM_D_DECL(i, j, k - 1), SimVar::momz) /
            U(KFVM_D_DECL(i, j, k - 1), SimVar::dens); // Naturally zero in 2d

    const Real divV = (dxU + dyV + dzW) / 2.0; // Undivided difference
    const Real kappa1 = 0.3,
               eta =
                   fmin(1.0, fmax(0.0, -(kappa1 * minSdSp + divV) / (kappa1 * minSdSp)));
    const Real kappa2 = kappa1 * (1.0 - eta);

    // Extend density and pressure ranges
    maxDens *= (1.0 + kappa2);
    minDens *= (1.0 - kappa2);
    minEint *= (1.0 - kappa2);

    const Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;

    // Enforce positive density
    // Find mixing amount
    Real theta = 1.0;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // West face
      Real rho = rsX(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      // theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
      //  East face
      rho = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      // theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
      //  South face
      rho = rsY(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      // theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
      //  North face
      rho = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      // theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
#if (SPACE_DIM == 3)
      // Bottom face
      rho = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
      // Top face
      rho = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, SimVar::dens);
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
                source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::dens);
            theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
            theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
          }
        }
      }
    }

    // Apply mixing
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

    // Stash theta before overwriting it
    wenoFlag(KFVM_D_DECL(i, j, k), 1) = theta;

    // Enforce positive internal energy
    // Find mixing amount
    theta = 1.0;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      ConsVec state;
      Real ei;
      // West face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      ei = EOS<EquationType::Hydro>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, eulerEintPos(minEint, refState, state));
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
            Real ei = EOS<EquationType::Hydro>::eint(state);
            theta = ei > minEint ? theta
                                 : fmin(theta, eulerEintPos(minEint, refState, state));
          }
        }
      }
    }

    // Update flag to most restrictive case
    wenoFlag(KFVM_D_DECL(i, j, k), 1) = fmin(theta, wenoFlag(KFVM_D_DECL(i, j, k), 1));

    // Apply mixing
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
  Real eulerEintPos(const Real emin, const Hydro::ConsVec refState,
                    const Hydro::ConsVec state) const {
    using Hydro::ConsVec;
    using Hydro::SimVar;
    using Kokkos::fabs;
    using Kokkos::fmin;

    Real a = 0.0, b = 1.0, c;
    Real dec = EOS<EquationType::Hydro>::eint(state) - emin;
    ConsVec stTheta;

    // Bisect on theta for 20 iterations regardless
    for (int nIter = 0; nIter < 20; nIter++) {
      const Real deb = dec; // dec is fval at b on entry

      // Midpoint and midpoint fval
      c = 0.5 * (a + b);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        stTheta[nV] = refState[nV] + c * (state[nV] - refState[nV]);
      }
      dec = EOS<EquationType::Hydro>::eint(stTheta) - emin;

      if (dec * deb < 0.0) {
        // Sign change in upper half, walk a forward
        a = c;
      } else {
        // default to lower half
        b = c;
      }
    }

    // Round down to lower bound to guarantee a valid state
    return a;
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
