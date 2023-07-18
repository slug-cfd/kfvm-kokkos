#ifndef HYDRO_K_H_
#define HYDRO_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits>
#include <math.h>

#include "../../Dimension.H"
#include "../../Geometry.H"
#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "../SimVar.H"
#include "Definitions.H"
#include "Hydro_EOS.H"
#include "Hydro_Riemann.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::Hydro> {
  const Real gamm, J;
  const Real rho, u, v, w, hvsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const Hydro::ConsVec ref_, Real gam_, Real J_)
      : gamm(gam_ - Real(1.0)), J(J_), rho(ref_[Hydro::SimVar::dens] / J),
        u(ref_[Hydro::SimVar::momx] / ref_[Hydro::SimVar::dens]),
        v(ref_[Hydro::SimVar::momy] / ref_[Hydro::SimVar::dens]),
        w(ref_[Hydro::SimVar::momz] / ref_[Hydro::SimVar::dens]),
        hvsq(Real(0.5) * (u * u + v * v + w * w)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {
    using Hydro::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) =
        gamm *
        (hvsq * state(SimVar::dens) - u * state(SimVar::momx) - v * state(SimVar::momy) -
         w * state(SimVar::momz) + state(SimVar::etot)) /
        J;

    state(SimVar::momx) = (state(SimVar::momx) - u * state(SimVar::dens)) / (J * rho);
    state(SimVar::momy) = (state(SimVar::momy) - v * state(SimVar::dens)) / (J * rho);
    state(SimVar::momz) = (state(SimVar::momz) - w * state(SimVar::dens)) / (J * rho);

    state(SimVar::dens) /= J;
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {
    using Hydro::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) =
        J * (hvsq * state(SimVar::dens) + rho * u * state(SimVar::momx) +
             rho * v * state(SimVar::momy) + rho * w * state(SimVar::momz) +
             state(SimVar::etot) / gamm);

    state(SimVar::momx) = J * (rho * state(SimVar::momx) + u * state(SimVar::dens));
    state(SimVar::momy) = J * (rho * state(SimVar::momy) + v * state(SimVar::dens));
    state(SimVar::momz) = J * (rho * state(SimVar::momz) + w * state(SimVar::dens));

    state(SimVar::dens) *= J;
  }
};

template <>
struct CompSpaceVars<EquationType::Hydro> {

  // Pass metric or inverse metric derivatives to go each direction
  KOKKOS_INLINE_FUNCTION static void convert(const Mat3 &mDeriv, Hydro::ConsVec state) {
    using Hydro::SimVar;

    const Real mXi = mDeriv.g11 * state[SimVar::momx] + mDeriv.g12 * state[SimVar::momy] +
                     mDeriv.g13 * state[SimVar::momz];
    const Real mEta = mDeriv.g21 * state[SimVar::momx] +
                      mDeriv.g22 * state[SimVar::momy] + mDeriv.g23 * state[SimVar::momz];
    const Real mZeta = mDeriv.g31 * state[SimVar::momx] +
                       mDeriv.g32 * state[SimVar::momy] +
                       mDeriv.g33 * state[SimVar::momz];

    state[SimVar::momx] = mXi;
    state[SimVar::momy] = mEta;
    state[SimVar::momz] = mZeta;
  }
};

template <class UViewType>
struct AuxVars<EquationType::Hydro, UViewType> {
  UViewType U;
  AuxDataView V;
  Geometry<geomType> geom;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const Geometry<geomType> &geom_,
          const EosParameters &eosParams_)
      : U(U_), V(V_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using Hydro::AuxVar;
    using Hydro::ConsVec;
    using Hydro::SimVar;

    // Get grid mapping Jacobian
    const Real J = geom.jacobian(KFVM_D_DECL(i, j, k));

    // Try not to assume the order of SimVars
    ConsVec ua;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      ua[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    V(KFVM_D_DECL(i, j, k), AuxVar::dens) = ua[SimVar::dens] / J;

    // Set velocities
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) = ua[SimVar::momx] / ua[SimVar::dens];
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) = ua[SimVar::momy] / ua[SimVar::dens];
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) = ua[SimVar::momz] / ua[SimVar::dens];

    // Internal energy and pressure
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) =
        EOS<EquationType::Hydro>::eint(J, ua) / V(KFVM_D_DECL(i, j, k), AuxVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) =
        EOS<EquationType::Hydro>::pres(eosParams.gamma, J, ua);
  }
};

template <RSType rst>
struct RiemannSolverX_K<EquationType::Hydro, rst> {
  FaceDataView U;
  Geometry<geomType> geom;
  EosParameters eosParams;

  RiemannSolverX_K(const FaceDataView &U_, const Geometry<geomType> &geom_,
                   const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // get metric terms
        const Real J = geom.jacobian(KFVM_D_DECL(i, j, k), KFVM_D_DECL(-0.5, nQ, nR));
        const Vec3 metric =
            geom.xiMetricDeriv(KFVM_D_DECL(i, j, k), KFVM_D_DECL(-0.5, nQ, nR));

        // pull out states and convert to primitive
        ConsVec uL, uR, F;
        PrimVec vL, vR;
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uL, vL);
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uR, vR);

        // Calculate flux
        Real vq =
            HydroRS<rst>::RiemannSolve_X(eosParams.gamma, metric, J, uL, uR, vL, vR, F);
        maxVel = vq > maxVel ? vq : maxVel;

        // Overwrite states with flux
        for (int nV = 0; nV < NUM_VARS; nV++) {
          U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
          U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
        }
      }
    }
  }
};

template <RSType rst>
struct RiemannSolverY_K<EquationType::Hydro, rst> {
  FaceDataView U;
  Geometry<geomType> geom;
  EosParameters eosParams;

  RiemannSolverY_K(const FaceDataView &U_, const Geometry<geomType> &geom_,
                   const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // get metric terms
        const Real J = geom.jacobian(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, -0.5, nR));
        const Vec3 metric =
            geom.etaMetricDeriv(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, -0.5, nR));

        // pull out states and convert to primitive
        ConsVec uL, uR, F;
        PrimVec vL, vR;
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uL, vL);
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uR, vR);

        // Calculate flux
        Real vq =
            HydroRS<rst>::RiemannSolve_Y(eosParams.gamma, metric, J, uL, uR, vL, vR, F);
        maxVel = vq > maxVel ? vq : maxVel;

        // Overwrite states with flux
        for (int nV = 0; nV < NUM_VARS; nV++) {
          U(KFVM_D_DECL(i, j, k), 0, nQ, nV) = F[nV];
          U(KFVM_D_DECL(i, j, k), 1, nQ, nV) = F[nV];
        }
      }
    }
  }
};

template <RSType rst>
struct RiemannSolverZ_K<EquationType::Hydro, rst> {
  FaceDataView U;
  Geometry<geomType> geom;
  EosParameters eosParams;

  RiemannSolverZ_K(const FaceDataView &U_, const Geometry<geomType> &geom_,
                   const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

#if (SPACE_DIM == 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k, Real &maxVel) const {
    using Hydro::ConsVec;
    using Hydro::PrimVec;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // get metric terms
        const Real J = geom.jacobian(i, j, k, nQ, nR, -0.5);
        const Vec3 metric = geom.zetaMetricDeriv(i, j, k, nQ, nR, -0.5);

        // pull out states and convert to primitive
        ConsVec uL, uR, F;
        PrimVec vL, vR;
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uL, vL);
        EOS<EquationType::Hydro>::cons2prim(eosParams.gamma, J, uR, vR);

        // Calculate flux
        Real vq =
            HydroRS<rst>::RiemannSolve_Z(eosParams.gamma, metric, J, uL, uR, vL, vR, F);
        maxVel = vq > maxVel ? vq : maxVel;

        // Overwrite states with flux
        for (int nV = 0; nV < NUM_VARS; nV++) {
          U(i, j, k, 0, nQ, nV) = F[nV];
          U(i, j, k, 1, nQ, nV) = F[nV];
        }
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
  Geometry<geomType> geom;
  Real t;

  SourceTerms_K(const SourceDataView &source_,
                KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                            const FaceDataView &rsZ_),
                const UViewType &U_, const WenoFlagView &wenoFlag_, const DiffMatView &D_,
                const QuadRuleView &ab_, const EosParameters &eosParams_,
                const Geometry<geomType> &geom_, const Real t_)
      : source(source_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), U(U_),
        wenoFlag(wenoFlag_), D(D_), ab(ab_), eosParams(eosParams_), geom(geom_), t(t_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;

    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
        for (int nS = 0; nS < nzQuad; nS++) {
          // Quad point in physical coordinates
          Vec4 xyzJ = geom.physCoordJac(KFVM_D_DECL(i, j, k),
                                        KFVM_D_DECL(ab(nQ), ab(nR), ab(nS)));

          Hydro::ConsVec state, force;
          // Real state[10],force[10];
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] =
                source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) / xyzJ.v4;
            force[nV] = 0;
          }

          // Fill force with user defined terms
          Hydro::SourceTerm(t, xyzJ.v1, xyzJ.v2, xyzJ.v3, state, force);

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
  Geometry<geomType> geom;
  EosParameters eosParams;

  PositivityPreserve_K(const UDataView &U_,
                       KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                                   const FaceDataView &rsZ_),
                       const bool haveSources_, const SourceDataView &source_,
                       const WenoFlagView &wenoFlag_, const Geometry<geomType> &geom_,
                       const EosParameters &eosParams_)
      : U(U_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), haveSources(haveSources_),
        source(source_), wenoFlag(wenoFlag_), geom(geom_), eosParams(eosParams_) {}

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
    const Real J = geom.jacobian(KFVM_D_DECL(i, j, k));
    Real rDens = refState[SimVar::dens] / J;
    Real rPres = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, refState);

    // Find ranges of density, pressure, and soundspeed
    Real maxDens = -std::numeric_limits<Real>::max();
    Real minDens = std::numeric_limits<Real>::max();
    Real minPres = std::numeric_limits<Real>::max();
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
          const Real Jo = geom.jacobian(KFVM_D_DECL(i + xo, j + yo, k + zo));
          maxDens = fmax(maxDens, state[SimVar::dens] / Jo);
          minDens = fmin(minDens, state[SimVar::dens] / Jo);
          Real pres = EOS<EquationType::Hydro>::pres(eosParams.gamma, Jo, state);
          Real sdsp = EOS<EquationType::Hydro>::soundspeed(eosParams.gamma, Jo, state);
          minPres = fmin(minPres, pres);
          minSdSp = fmin(minSdSp, sdsp);
#if (SPACE_DIM == 3)
        }
#endif
      }
    }

    // Find size of flattener
    Real dxU = U(KFVM_D_DECL(i + 1, j, k), SimVar::momx) /
                   U(KFVM_D_DECL(i + 1, j, k), SimVar::dens) -
               U(KFVM_D_DECL(i - 1, j, k), SimVar::momx) /
                   U(KFVM_D_DECL(i - 1, j, k), SimVar::dens);
    Real dyV = U(KFVM_D_DECL(i, j + 1, k), SimVar::momy) /
                   U(KFVM_D_DECL(i, j + 1, k), SimVar::dens) -
               U(KFVM_D_DECL(i, j - 1, k), SimVar::momy) /
                   U(KFVM_D_DECL(i, j - 1, k), SimVar::dens);
#if (SPACE_DIM == 3)
    Real dzW = U(KFVM_D_DECL(i, j, k + 1), SimVar::momz) /
                   U(KFVM_D_DECL(i, j, k + 1), SimVar::dens) -
               U(KFVM_D_DECL(i, j, k - 1), SimVar::momz) /
                   U(KFVM_D_DECL(i, j, k - 1), SimVar::dens);
#else
    Real dzW = Real(0.0);
#endif
    Real divV = (dxU + dyV + dzW) / Real(2.0); // Undivided difference
    Real kappa1 = Real(0.3), kappa2 = Real(0.3);
    Real eta =
        fmin(Real(1.0), fmax(Real(0.0), -(kappa1 * minSdSp + divV) / (kappa1 * minSdSp)));

    // Extend density and pressure ranges
    maxDens *= (Real(1.0) + kappa2 - kappa2 * eta);
    minDens *= (Real(1.0) - kappa2 + kappa2 * eta);
    minPres *= (Real(1.0) - kappa2 + kappa2 * eta);

    Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;

    wenoFlag(KFVM_D_DECL(i, j, k), 1) = 1.0;
    return;

    // Enforce positive density
    // Find mixing amount
    Real theta = Real(1.0);
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // West face
      Real rho = rsX(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // East face
      rho = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, SimVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // South face
      rho = rsY(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // North face
      rho = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, SimVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
#if (SPACE_DIM == 3)
      // Bottom face
      rho = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho < minDens ? fmin(theta, rDensDiffM / (rDens - rho)) : theta;
      theta = rho > maxDens ? fmin(theta, rDensDiffP / (rho - rDens)) : theta;
      // Top face
      rho = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, SimVar::dens);
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
                source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::dens);
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

    return;

    // Enforce positive pressure
    // Find mixing amount
    theta = Real(1.0);
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      ConsVec state;
      Real p;
      // West face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
      theta = p >= minPres
                  ? theta
                  : fmin(theta, eulerPresPos(eosParams.gamma, minPres, refState, state));
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
            Real p = EOS<EquationType::Hydro>::pres(eosParams.gamma, J, state);
            theta = p >= minPres ? theta
                                 : fmin(theta, eulerPresPos(eosParams.gamma, minPres,
                                                            refState, state));
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
  Real eulerPresPos(const Real gamma, const Real pmin, const Hydro::ConsVec refState,
                    const Hydro::ConsVec state) const {
    using Hydro::SimVar;
    using Kokkos::sqrt;

    Real emin = pmin / (gamma - Real(1.0));

    Real A = Real(2.0) * (state[SimVar::dens] - refState[SimVar::dens]) *
             (state[SimVar::etot] - refState[SimVar::etot]);
    A -= ((state[SimVar::momx] - refState[SimVar::momx]) *
              (state[SimVar::momx] - refState[SimVar::momx]) +
          (state[SimVar::momy] - refState[SimVar::momy]) *
              (state[SimVar::momy] - refState[SimVar::momy]) +
          (state[SimVar::momz] - refState[SimVar::momz]) *
              (state[SimVar::momz] - refState[SimVar::momz]));

    Real B = Real(2.0) *
             (refState[SimVar::dens] * (state[SimVar::etot] - refState[SimVar::etot]) +
              refState[SimVar::etot] * (state[SimVar::dens] - refState[SimVar::dens]));
    B -= Real(2.0) *
         ((state[SimVar::momx] - refState[SimVar::momx]) * refState[SimVar::momx] +
          (state[SimVar::momy] - refState[SimVar::momy]) * refState[SimVar::momy] +
          (state[SimVar::momz] - refState[SimVar::momz]) * refState[SimVar::momz]);
    B -= Real(2.0) * emin * (state[SimVar::dens] - refState[SimVar::dens]);

    Real C = Real(2.0) * refState[SimVar::dens] * (refState[SimVar::etot] - emin);
    C -= (refState[SimVar::momx] * refState[SimVar::momx] +
          refState[SimVar::momy] * refState[SimVar::momy] +
          refState[SimVar::momz] * refState[SimVar::momz]);

    Real rtdisc = sqrt(B * B - Real(4.0) * A * C);
    Real th1 = (-B + rtdisc) / (Real(2.0) * A), th2 = (-B - rtdisc) / (Real(2.0) * A);

    if (Real(0.0) <= th1 && th1 <= Real(1.0) && Real(0.0) <= th2 && th2 <= Real(1.0)) {
      return th1 < th2 ? th1 : th2;
    } else if (Real(0.0) <= th1 && th1 <= Real(1.0)) {
      return th1;
    } else if (Real(0.0) <= th2 && th2 <= Real(1.0)) {
      return th2;
    } else {
      return Real(0.0);
    }
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
