#ifndef MHD_GLM_K_H_
#define MHD_GLM_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits.h>

#include "../../Dimension.H"
#include "../../Geometry.H"
#include "../../ParameterStructs.H"
#include "../../Types.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "../SimVar.H"
#include "Definitions.H"
#include "MHD_GLM_EOS.H"
#include "MHD_GLM_Riemann.H"
#include "Numeric.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::MHD_GLM> {
  const Real gamm;
  const Real rho, u, v, w, bx, by, bz, rPsi, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const MHD_GLM::ConsVec ref_, Real gam_)
      : gamm(gam_ - Real(1.0)), rho(ref_[MHD_GLM::SimVar::dens]),
        u(ref_[MHD_GLM::SimVar::momx] / rho), v(ref_[MHD_GLM::SimVar::momy] / rho),
        w(ref_[MHD_GLM::SimVar::momz] / rho), bx(ref_[MHD_GLM::SimVar::magx]),
        by(ref_[MHD_GLM::SimVar::magy]), bz(ref_[MHD_GLM::SimVar::magz]),
        rPsi(ref_[MHD_GLM::SimVar::psi]), vsq((u * u + v * v + w * w) / Real(2.0)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {
    using MHD_GLM::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) =
        gamm *
        (vsq * state(SimVar::dens) + state(SimVar::etot) - u * state(SimVar::momx) -
         v * state(SimVar::momy) - w * state(SimVar::momz) - bx * state(SimVar::magx) -
         by * state(SimVar::magy) - bz * state(SimVar::magz) - rPsi * state(SimVar::psi));

    state(SimVar::momx) = (state(SimVar::momx) - u * state(SimVar::dens)) / rho;
    state(SimVar::momy) = (state(SimVar::momy) - v * state(SimVar::dens)) / rho;
    state(SimVar::momz) = (state(SimVar::momz) - w * state(SimVar::dens)) / rho;
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {
    using MHD_GLM::SimVar;

    // Doing etot first allows state to be overwritten safely
    state(SimVar::etot) =
        (vsq * state(SimVar::dens) + rho * u * state(SimVar::momx) +
         rho * v * state(SimVar::momy) + rho * w * state(SimVar::momz) +
         bx * state(SimVar::magx) + by * state(SimVar::magy) + bz * state(SimVar::magz) +
         state(SimVar::etot) / gamm + rPsi * state(SimVar::psi));

    state(SimVar::momx) = (rho * state(SimVar::momx) + u * state(SimVar::dens));
    state(SimVar::momy) = (rho * state(SimVar::momy) + v * state(SimVar::dens));
    state(SimVar::momz) = (rho * state(SimVar::momz) + w * state(SimVar::dens));
  }
};

template <class UViewType>
struct AuxVars<EquationType::MHD_GLM, UViewType> {
  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using MHD_GLM::AuxVar;
    using MHD_GLM::ConsVec;
    using MHD_GLM::SimVar;

    // Try not to assume the order of SimVars
    ConsVec u;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      u[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Set velocities
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) =
        U(KFVM_D_DECL(i, j, k), SimVar::momx) / U(KFVM_D_DECL(i, j, k), SimVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) =
        U(KFVM_D_DECL(i, j, k), SimVar::momy) / U(KFVM_D_DECL(i, j, k), SimVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) =
        U(KFVM_D_DECL(i, j, k), SimVar::momz) / U(KFVM_D_DECL(i, j, k), SimVar::dens);

    // Internal energy and pressures
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(u);
    V(KFVM_D_DECL(i, j, k), AuxVar::prsg) =
        EOS<EquationType::MHD_GLM>::prsg(eosParams.gamma, u);
    V(KFVM_D_DECL(i, j, k), AuxVar::prsb) = EOS<EquationType::MHD_GLM>::prsb(u);
  }
};

template <class UViewType>
struct FlowStats<EquationType::MHD_GLM, UViewType> {
  UViewType U;
  Geometry geom;
  EosParameters eosParams;

  FlowStats(const UViewType &U_, const Geometry &geom_, const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  FlowStatsArray &stat) const {
    using MHD_GLM::SimVar;
    using MHD_GLM::StatVar;

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

    // Compute mag field gradients
    const Real dybx = 0.5 *
                      (U(KFVM_D_DECL(i, j + 1, k), SimVar::magx) -
                       U(KFVM_D_DECL(i, j - 1, k), SimVar::magx)) /
                      geom.dy;
    const Real dxby = 0.5 *
                      (U(KFVM_D_DECL(i + 1, j, k), SimVar::magy) -
                       U(KFVM_D_DECL(i - 1, j, k), SimVar::magy)) /
                      geom.dx;
#if (SPACE_DIM == 2)
    const Real dzbx = 0.0, dzby = 0.0, dxbz = 0.0, dybz = 0.0;
#else
    const Real dzbx =
        0.5 * (U(i, j, k + 1, SimVar::magx) - U(i, j, k - 1, SimVar::magx)) / geom.dz;

    const Real dzby =
        0.5 * (U(i, j, k + 1, SimVar::magy) - U(i, j, k - 1, SimVar::magy)) / geom.dz;

    const Real dxbz =
        0.5 * (U(i + 1, j, k, SimVar::magz) - U(i - 1, j, k, SimVar::magz)) / geom.dx;

    const Real dybz =
        0.5 * (U(i, j + 1, k, SimVar::magz) - U(i, j - 1, k, SimVar::magz)) / geom.dy;
#endif

    // compute vorticity and enstrophy
    const Real cbx = dybz - dzby, cby = dzbx - dxbz, cbz = dxby - dybx;
    const Real bCrossCb = U(KFVM_D_DECL(i, j, k), SimVar::magx) * cbx +
                          U(KFVM_D_DECL(i, j, k), SimVar::magy) * cby +
                          U(KFVM_D_DECL(i, j, k), SimVar::magz) * cbz;

    // Add into array for reduction
    const Real ifac = geom.dvol / geom.volume;
    stat.data[StatVar::ekin] += ifac * ek;
    stat.data[StatVar::enst] += ifac * et;
    stat.data[StatVar::heli] += ifac * bCrossCb;
  }
};

template <RSType rst>
struct RiemannSolverX_K<EquationType::MHD_GLM, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverX_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using MHD_GLM::ConsVec;
    using MHD_GLM::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = MHD_GLM_RS<rst>::RiemannSolve_X(eosParams.gamma, eosParams.ch_glm, uL, uR,
                                                vL, vR, F);
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
struct RiemannSolverY_K<EquationType::MHD_GLM, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverY_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using MHD_GLM::ConsVec;
    using MHD_GLM::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = MHD_GLM_RS<rst>::RiemannSolve_Y(eosParams.gamma, eosParams.ch_glm, uL, uR,
                                                vL, vR, F);
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
struct RiemannSolverZ_K<EquationType::MHD_GLM, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverZ_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

#if (SPACE_DIM == 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k, Real &maxVel) const {
    using MHD_GLM::ConsVec;
    using MHD_GLM::PrimVec;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // Pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::MHD_GLM>::cons2prim(eosParams.gamma, uR, vR);

      // Calculate flux
      Real vq = MHD_GLM_RS<rst>::RiemannSolve_Z(eosParams.gamma, eosParams.ch_glm, uL, uR,
                                                vL, vR, F);
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
struct SpeedEstimate_K<EquationType::MHD_GLM> {
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  EosParameters eosParams;

  SpeedEstimate_K(KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                              const FaceDataView &rsZ_),
                  const EosParameters &eosParams_)
      : KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &maxLam,
                  Real &maxVel) const {
    using Kokkos::fabs;
    using MHD_GLM::ConsVec;
    using MHD_GLM::SimVar;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    // Check x-faces
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states
      Real cf, lam, vel;
      ConsVec uM, uP;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uM[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
        uP[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }

      // Minus face
      vel = fabs(uM[SimVar::momx] / uM[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uM);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;

      // Plus face
      vel = fabs(uP[SimVar::momx] / uP[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uP);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;
    }

    // Check y-faces
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states
      Real cf, lam, vel;
      ConsVec uM, uP;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uM[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
        uP[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }

      // Minus face
      vel = fabs(uM[SimVar::momx] / uM[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uM);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;

      // Plus face
      vel = fabs(uP[SimVar::momx] / uP[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uP);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;
    }

#if (SPACE_DIM == 3)
    // Check z-faces
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states
      Real cf, lam, vel;
      ConsVec uM, uP;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uM[nV] = rsZ(i, j, k, 1, nQ, nV);
        uP[nV] = rsZ(i, j, k + 1, 0, nQ, nV);
      }

      // Minus face
      vel = fabs(uM[SimVar::momx] / uM[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uM);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;

      // Plus face
      vel = fabs(uP[SimVar::momx] / uP[SimVar::dens]);
      cf = EOS<EquationType::MHD_GLM>::fastMagSonicX(eosParams.gamma, uP);
      lam = cf + vel;
      maxVel = vel > maxVel ? vel : maxVel;
      maxLam = lam > maxLam ? lam : maxLam;
    }
#endif
  }
};

namespace MHD_GLM {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::MHD_GLM, UViewType> {
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
    using Kokkos::exp;
    using MHD_GLM::SimVar;

    const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;

    // Get cell center coordinate
    const Real xc = geom.xCoord(i);
    const Real yc = geom.yCoord(j);
#if (SPACE_DIM == 2)
    const Real zc = geom.zCoord(0);
#else
    const Real zc = geom.zCoord(k);
#endif

    // get subviews of point values for this cell
    auto cellVals =
        Kokkos::subview(source, KFVM_D_DECL(i, j, k),
                        KFVM_D_DECL(Kokkos::ALL, Kokkos::ALL, Kokkos::ALL), Kokkos::ALL);
    auto wFaceVals =
        Kokkos::subview(rsX, KFVM_D_DECL(i, j, k), 1, Kokkos::ALL, Kokkos::ALL);
    auto eFaceVals =
        Kokkos::subview(rsX, KFVM_D_DECL(i + 1, j, k), 0, Kokkos::ALL, Kokkos::ALL);
    auto sFaceVals =
        Kokkos::subview(rsY, KFVM_D_DECL(i, j, k), 1, Kokkos::ALL, Kokkos::ALL);
    auto nFaceVals =
        Kokkos::subview(rsY, KFVM_D_DECL(i, j + 1, k), 0, Kokkos::ALL, Kokkos::ALL);
#if (SPACE_DIM == 3)
    auto bFaceVals =
        Kokkos::subview(rsZ, KFVM_D_DECL(i, j, k), 1, Kokkos::ALL, Kokkos::ALL);
    auto tFaceVals =
        Kokkos::subview(rsZ, KFVM_D_DECL(i, j, k + 1), 0, Kokkos::ALL, Kokkos::ALL);
#endif

    // Divergence and psi gradient
    Real divB = 0.0;
#if (SPACE_DIM == 2)
    Real gpsiDu[NUM_QUAD_PTS][NUM_QUAD_PTS][1];
#else
    Real gpsiDu[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_QUAD_PTS];
#endif

    // B field divergence
    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (int nS = 0; nS < nzQuad; nS++) {
        // Flattened index and total quadrature weight
        const int qIdx = nS * NUM_QUAD_PTS + nQ;
        const Real twt = SPACE_DIM == 2 ? wt(nQ) : wt(nQ) * wt(nS);

        // Bx normal on E/W faces
        const Real bAvW = 0.5 * (rsX(KFVM_D_DECL(i, j, k), 1, qIdx, SimVar::magx) +
                                 rsX(KFVM_D_DECL(i, j, k), 0, qIdx, SimVar::magx));
        const Real bAvE = 0.5 * (rsX(KFVM_D_DECL(i + 1, j, k), 0, qIdx, SimVar::magx) +
                                 rsX(KFVM_D_DECL(i + 1, j, k), 1, qIdx, SimVar::magx));
        divB += twt * (bAvE - bAvW) / geom.dx;
        // By normal on N/S faces
        const Real bAvS = 0.5 * (rsY(KFVM_D_DECL(i, j, k), 1, qIdx, SimVar::magy) +
                                 rsY(KFVM_D_DECL(i, j, k), 0, qIdx, SimVar::magy));
        const Real bAvN = 0.5 * (rsY(KFVM_D_DECL(i, j + 1, k), 0, qIdx, SimVar::magy) +
                                 rsY(KFVM_D_DECL(i, j + 1, k), 1, qIdx, SimVar::magy));
        divB += twt * (bAvN - bAvS) / geom.dy;
#if (SPACE_DIM == 3)
        // By normal on N/S faces
        const Real bAvB = 0.5 * (rsZ(i, j, k, 1, qIdx, SimVar::magz) +
                                 rsZ(i, j, k, 0, qIdx, SimVar::magz));
        const Real bAvT = 0.5 * (rsZ(i, j, k + 1, 0, qIdx, SimVar::magz) +
                                 rsZ(i, j, k + 1, 1, qIdx, SimVar::magz));
        divB += twt * (bAvT - bAvB) / geom.dz;
#endif
      }
    }

    // Find psi gradient terms
    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
        for (int nS = 0; nS < nzQuad; nS++) {
          // Point values of velocity for gpsiDu
          const Real rho = cellVals(KFVM_D_DECL(nQ, nR, nS), SimVar::dens);
          const Real vx = cellVals(KFVM_D_DECL(nQ, nR, nS), SimVar::momx) / rho;
          const Real vy = cellVals(KFVM_D_DECL(nQ, nR, nS), SimVar::momy) / rho;
          const Real vz = cellVals(KFVM_D_DECL(nQ, nR, nS), SimVar::momz) / rho;

          // Derivatives of B and psi, fall back on centered fd if PP corrected too far
          Real dpx, dpy, dpz = 0.0;
          if (wenoFlag(KFVM_D_DECL(i, j, k), 1) < 0.75) {
            const Real tdx = 1.0 / (2.0 * geom.dx), tdy = 1.0 / (2.0 * geom.dy);
            dpx = tdx * (U(KFVM_D_DECL(i + 1, j, k), SimVar::psi) -
                         U(KFVM_D_DECL(i - 1, j, k), SimVar::psi));
            dpy = tdy * (U(KFVM_D_DECL(i, j + 1, k), SimVar::psi) -
                         U(KFVM_D_DECL(i, j - 1, k), SimVar::psi));
#if (SPACE_DIM == 3)
            const Real tdz = 1.0 / (2.0 * geom.dz);
            dpz = tdz * (U(i, j, k + 1, SimVar::psi) - U(i, j, k - 1, SimVar::psi));
#endif
          } else {
            dpx = diffX(cellVals, wFaceVals, eFaceVals, nQ, nR, nS, SimVar::psi);
            dpy = diffY(cellVals, sFaceVals, nFaceVals, nQ, nR, nS, SimVar::psi);
#if (SPACE_DIM == 3)
            dpz = diffZ(cellVals, bFaceVals, tFaceVals, nQ, nR, nS, SimVar::psi);
#endif
          }
          gpsiDu[nQ][nR][nS] = vx * dpx + vy * dpy + vz * dpz;
        }
      }
    }

    // Find and fill forcing
    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      Real x = Numeric::sclAb(xc, geom.dx, ab(nQ));
      for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
        Real y = Numeric::sclAb(yc, geom.dy, ab(nR));
        for (int nS = 0; nS < nzQuad; nS++) {
          Real z = SPACE_DIM == 2 ? zc : Numeric::sclAb(zc, geom.dz, ab(nS));
          MHD_GLM::ConsVec state, force;
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            force[nV] = 0;
          }

          // Fill force with user defined terms
          MHD_GLM::SourceTerm(t, x, y, z, eosParams, userParams, state, force);

          // Define GLM source terms
          const Real vx = state[SimVar::momx] / state[SimVar::dens];
          const Real vy = state[SimVar::momy] / state[SimVar::dens];
          const Real vz = state[SimVar::momz] / state[SimVar::dens];

          const Real uDb = vx * state[SimVar::magx] + vy * state[SimVar::magy] +
                           vz * state[SimVar::magz];

          // Fill source view with GLM and user forcing
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::dens) =
              force[SimVar::dens];

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::momx) =
              force[SimVar::momx] - divB * state[SimVar::magx];
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::momy) =
              force[SimVar::momy] - divB * state[SimVar::magy];
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::momz) =
              force[SimVar::momz] - divB * state[SimVar::magz];

          Real eForce = divB * uDb + gpsiDu[nQ][nR][nS] * state[SimVar::psi];
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::etot) =
              force[SimVar::etot] - eForce;

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::magx) =
              force[SimVar::magx] - divB * vx;
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::magy) =
              force[SimVar::magy] - divB * vy;
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::magz) =
              force[SimVar::magz] - divB * vz;

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), SimVar::psi) =
              -gpsiDu[nQ][nR][nS];
        }
      }
    }
  }

  template <class CellView, class MFaceView, class PFaceView>
  KOKKOS_INLINE_FUNCTION Real diffX(CellView &cellVals, const MFaceView &mFace,
                                    const PFaceView &pFace, const int nQ, const int nR,
                                    const int nS, const int nV) const {
    // Evaluate boundary terms
    int faceIdx = SPACE_DIM == 2 ? nR : nR * NUM_QUAD_PTS + nS;
    Real dvdx =
        D(nQ, 0) * mFace(faceIdx, nV) + D(nQ, NUM_QUAD_PTS + 1) * pFace(faceIdx, nV);
    for (int n = 1; n <= NUM_QUAD_PTS; n++) {
      dvdx += D(nQ, n) * cellVals(KFVM_D_DECL(n - 1, nR, nS), nV);
    }
    dvdx /= geom.dx;
    return dvdx;
  }

  template <class CellView, class MFaceView, class PFaceView>
  KOKKOS_INLINE_FUNCTION Real diffY(CellView &cellVals, const MFaceView &mFace,
                                    const PFaceView &pFace, const int nQ, const int nR,
                                    const int nS, const int nV) const {
    // Evaluate boundary terms
    int faceIdx = SPACE_DIM == 2 ? nQ : nQ * NUM_QUAD_PTS + nS;
    Real dvdy =
        D(nR, 0) * mFace(faceIdx, nV) + D(nR, NUM_QUAD_PTS + 1) * pFace(faceIdx, nV);
    for (int n = 1; n <= NUM_QUAD_PTS; n++) {
      dvdy += D(nR, n) * cellVals(KFVM_D_DECL(nQ, n - 1, nS), nV);
    }
    dvdy /= geom.dy;
    return dvdy;
  }

  template <class CellView, class MFaceView, class PFaceView>
  KOKKOS_INLINE_FUNCTION Real diffZ(CellView &cellVals, const MFaceView &mFace,
                                    const PFaceView &pFace, const int nQ, const int nR,
                                    const int nS, const int nV) const {
    // Evaluate boundary terms
    int faceIdx = nQ * NUM_QUAD_PTS + nR;
    Real dvdz =
        D(nS, 0) * mFace(faceIdx, nV) + D(nS, NUM_QUAD_PTS + 1) * pFace(faceIdx, nV);
    for (int n = 1; n <= NUM_QUAD_PTS; n++) {
      dvdz += D(nS, n) * cellVals(KFVM_D_DECL(nQ, nR, n - 1), nV);
    }
    dvdz /= geom.dz;
    return dvdz;
  }
};

template <class UDataView>
struct PositivityPreserve_K<EquationType::MHD_GLM, UDataView> {
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
    using MHD_GLM::ConsVec;
    using MHD_GLM::SimVar;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    // Save reference state
    ConsVec refState;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }
    const Real rDens = refState[SimVar::dens];
    const Real rEint = EOS<EquationType::MHD_GLM>::eint(refState);

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
          Real eint = EOS<EquationType::MHD_GLM>::eint(state);
          Real sdsp = EOS<EquationType::MHD_GLM>::soundspeed(eosParams.gamma, state);
          minEint = fmin(minEint, eint);
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
    const Real kappa1 = 0.3,
               eta =
                   fmin(1.0, fmax(0.0, -(kappa1 * minSdSp + divV) / (kappa1 * minSdSp)));
    const Real kappa2 = kappa1 * (1.0 - eta);

    // Extend density and pressure ranges
    maxDens *= (1.0 + kappa2);
    minDens *= (1.0 - kappa2);
    minEint *= (1.0 - kappa2);

    Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;

    // Enforce positive density
    // Find mixing amount given by face values
    Real theta = 1.0;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // West face
      Real rho = rsX(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  East face
      rho = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  South face
      rho = rsY(KFVM_D_DECL(i, j, k), 1, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));

      //  North face
      rho = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, SimVar::dens);
      theta = rho > minDens ? theta : fmin(theta, rDensDiffM / (rDens - rho));
      theta = rho < maxDens ? theta : fmin(theta, rDensDiffP / (rho - rDens));
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
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));

      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));

      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));

      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));

      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      ei = EOS<EquationType::MHD_GLM>::eint(state);
      theta = ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));
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
            Real ei = EOS<EquationType::MHD_GLM>::eint(state);
            theta =
                ei > minEint ? theta : fmin(theta, mhdEintPos(minEint, refState, state));
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
  Real mhdEintPos(const Real emin, const MHD_GLM::ConsVec refState,
                  const MHD_GLM::ConsVec state) const {
    using Kokkos::fabs;
    using Kokkos::fmin;
    using MHD_GLM::ConsVec;
    using MHD_GLM::SimVar;

    Real a = 0.0, b = 1.0, c;
    Real dec = EOS<EquationType::MHD_GLM>::eint(state) - emin;
    ConsVec stTheta;

    // Bisect on theta for 20 iterations regardless
    for (int nIter = 0; nIter < 20; nIter++) {
      const Real deb = dec; // dec is fval at b on entry

      // Midpoint and midpoint fval
      c = 0.5 * (a + b);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        stTheta[nV] = refState[nV] + c * (state[nV] - refState[nV]);
      }
      dec = EOS<EquationType::MHD_GLM>::eint(stTheta) - emin;

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
