#ifndef MHD_8W_K_H_
#define MHD_8W_K_H_

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
#include "MHD_8W_EOS.H"
#include "Numeric.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::MHD_8W> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;

  const Real gamm;
  const Real rho, u, v, w, bx, by, bz, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const ConsVec ref_, Real gam_)
      : gamm(gam_ - Real(1.0)), rho(ref_[ConsVar::dens]), u(ref_[ConsVar::momx] / rho),
        v(ref_[ConsVar::momy] / rho), w(ref_[ConsVar::momz] / rho),
        bx(ref_[ConsVar::magx]), by(ref_[ConsVar::magy]), bz(ref_[ConsVar::magz]),
        vsq(0.5 * (u * u + v * v + w * w)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {

    // Doing etot first allows state to be overwritten safely
    state(ConsVar::etot) = gamm * (vsq * state(ConsVar::dens) + state(ConsVar::etot) -
                                   u * state(ConsVar::momx) - v * state(ConsVar::momy) -
                                   w * state(ConsVar::momz) - bx * state(ConsVar::magx) -
                                   by * state(ConsVar::magy) - bz * state(ConsVar::magz));

    state(ConsVar::momx) = (state(ConsVar::momx) - u * state(ConsVar::dens)) / rho;
    state(ConsVar::momy) = (state(ConsVar::momy) - v * state(ConsVar::dens)) / rho;
    state(ConsVar::momz) = (state(ConsVar::momz) - w * state(ConsVar::dens)) / rho;
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {

    // Doing etot first allows state to be overwritten safely
    state(ConsVar::etot) =
        (vsq * state(ConsVar::dens) + rho * u * state(ConsVar::momx) +
         rho * v * state(ConsVar::momy) + rho * w * state(ConsVar::momz) +
         bx * state(ConsVar::magx) + by * state(ConsVar::magy) +
         bz * state(ConsVar::magz) + state(ConsVar::etot) / gamm);

    state(ConsVar::momx) = (rho * state(ConsVar::momx) + u * state(ConsVar::dens));
    state(ConsVar::momy) = (rho * state(ConsVar::momy) + v * state(ConsVar::dens));
    state(ConsVar::momz) = (rho * state(ConsVar::momz) + w * state(ConsVar::dens));
  }
};

template <class UViewType>
struct AuxVars<EquationType::MHD_8W, UViewType> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using AuxVar = typename SimVar<EquationType::MHD_8W>::AuxVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;

  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Try not to assume the order of SimVars
    ConsVec u;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      u[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Set velocities
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momx) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momy) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momz) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);

    // Internal energy and pressures
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::MHD_8W>::eint(u);
    V(KFVM_D_DECL(i, j, k), AuxVar::prsg) =
        EOS<EquationType::MHD_8W>::prsg(eosParams.gamma, u);
    V(KFVM_D_DECL(i, j, k), AuxVar::prsb) = EOS<EquationType::MHD_8W>::prsb(u);
  }
};

template <class UViewType>
struct FlowStats<EquationType::MHD_8W, UViewType> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using StatVar = typename SimVar<EquationType::MHD_8W>::StatVar;

  UViewType U;
  Geometry geom;
  EosParameters eosParams;

  FlowStats(const UViewType &U_, const Geometry &geom_, const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  FlowStatsArray &stat) const {

    // Compute kinetic energy
    const Real ek =
        0.5 *
        (U(KFVM_D_DECL(i, j, k), ConsVar::momx) * U(KFVM_D_DECL(i, j, k), ConsVar::momx) +
         U(KFVM_D_DECL(i, j, k), ConsVar::momy) * U(KFVM_D_DECL(i, j, k), ConsVar::momy) +
         U(KFVM_D_DECL(i, j, k), ConsVar::momz) *
             U(KFVM_D_DECL(i, j, k), ConsVar::momz)) /
        U(KFVM_D_DECL(i, j, k), ConsVar::dens);

    // Compute velocity gradients
    const Real dyu = 0.5 *
                     (U(KFVM_D_DECL(i, j + 1, k), ConsVar::momx) /
                          U(KFVM_D_DECL(i, j + 1, k), ConsVar::dens) -
                      U(KFVM_D_DECL(i, j - 1, k), ConsVar::momx) /
                          U(KFVM_D_DECL(i, j - 1, k), ConsVar::dens)) /
                     geom.dy;
    const Real dxv = 0.5 *
                     (U(KFVM_D_DECL(i + 1, j, k), ConsVar::momy) /
                          U(KFVM_D_DECL(i + 1, j, k), ConsVar::dens) -
                      U(KFVM_D_DECL(i - 1, j, k), ConsVar::momy) /
                          U(KFVM_D_DECL(i - 1, j, k), ConsVar::dens)) /
                     geom.dx;
#if (SPACE_DIM == 2)
    const Real dzu = 0.0, dzv = 0.0, dxw = 0.0, dyw = 0.0;
#else
    const Real dzu = 0.5 *
                     (U(i, j, k + 1, ConsVar::momx) / U(i, j, k + 1, ConsVar::dens) -
                      U(i, j, k - 1, ConsVar::momx) / U(i, j, k - 1, ConsVar::dens)) /
                     geom.dz;

    const Real dzv = 0.5 *
                     (U(i, j, k + 1, ConsVar::momy) / U(i, j, k + 1, ConsVar::dens) -
                      U(i, j, k - 1, ConsVar::momy) / U(i, j, k - 1, ConsVar::dens)) /
                     geom.dz;

    const Real dxw = 0.5 *
                     (U(i + 1, j, k, ConsVar::momz) / U(i + 1, j, k, ConsVar::dens) -
                      U(i - 1, j, k, ConsVar::momz) / U(i - 1, j, k, ConsVar::dens)) /
                     geom.dx;

    const Real dyw = 0.5 *
                     (U(i, j + 1, k, ConsVar::momz) / U(i, j + 1, k, ConsVar::dens) -
                      U(i, j - 1, k, ConsVar::momz) / U(i, j - 1, k, ConsVar::dens)) /
                     geom.dy;
#endif

    // compute vorticity and enstrophy
    const Real omx = dyw - dzv, omy = dzu - dxw, omz = dxv - dyu;
    const Real et = 0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) *
                    (omx * omx + omy * omy + omz * omz);

    // Compute mag field gradients
    const Real dybx = 0.5 *
                      (U(KFVM_D_DECL(i, j + 1, k), ConsVar::magx) -
                       U(KFVM_D_DECL(i, j - 1, k), ConsVar::magx)) /
                      geom.dy;
    const Real dxby = 0.5 *
                      (U(KFVM_D_DECL(i + 1, j, k), ConsVar::magy) -
                       U(KFVM_D_DECL(i - 1, j, k), ConsVar::magy)) /
                      geom.dx;
#if (SPACE_DIM == 2)
    const Real dzbx = 0.0, dzby = 0.0, dxbz = 0.0, dybz = 0.0;
#else
    const Real dzbx =
        0.5 * (U(i, j, k + 1, ConsVar::magx) - U(i, j, k - 1, ConsVar::magx)) / geom.dz;

    const Real dzby =
        0.5 * (U(i, j, k + 1, ConsVar::magy) - U(i, j, k - 1, ConsVar::magy)) / geom.dz;

    const Real dxbz =
        0.5 * (U(i + 1, j, k, ConsVar::magz) - U(i - 1, j, k, ConsVar::magz)) / geom.dx;

    const Real dybz =
        0.5 * (U(i, j + 1, k, ConsVar::magz) - U(i, j - 1, k, ConsVar::magz)) / geom.dy;
#endif

    // compute vorticity and enstrophy
    const Real cbx = dybz - dzby, cby = dzbx - dxbz, cbz = dxby - dybx;
    const Real bCrossCb = U(KFVM_D_DECL(i, j, k), ConsVar::magx) * cbx +
                          U(KFVM_D_DECL(i, j, k), ConsVar::magy) * cby +
                          U(KFVM_D_DECL(i, j, k), ConsVar::magz) * cbz;

    // Add into array for reduction
    const Real ifac = geom.dvol / geom.volume;
    stat.data[StatVar::ekin] += ifac * ek;
    stat.data[StatVar::enst] += ifac * et;
    stat.data[StatVar::heli] += ifac * bCrossCb;
  }
};

template <>
struct SpeedEstimate_K<EquationType::MHD_8W> {
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

namespace MHD_8W {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::MHD_8W, UViewType> {
  using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;
  using ConsVec = typename SimVar<EquationType::MHD_8W>::ConsVec;

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
    using ConsVar = typename SimVar<EquationType::MHD_8W>::ConsVar;

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

    Real divB = 0.0;
    for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (int nS = 0; nS < nzQuad; nS++) {
        // Flattened index and total quadrature weight
        const int qIdx = nS * NUM_QUAD_PTS + nQ;
        const Real twt = SPACE_DIM == 2 ? wt(nQ) : wt(nQ) * wt(nS);

        // Bx normal on E/W faces
        const Real bAvW = 0.5 * (rsX(KFVM_D_DECL(i, j, k), 1, qIdx, ConsVar::magx) +
                                 rsX(KFVM_D_DECL(i, j, k), 0, qIdx, ConsVar::magx));
        const Real bAvE = 0.5 * (rsX(KFVM_D_DECL(i + 1, j, k), 0, qIdx, ConsVar::magx) +
                                 rsX(KFVM_D_DECL(i + 1, j, k), 1, qIdx, ConsVar::magx));
        divB += twt * (bAvE - bAvW) / geom.dx;
        // By normal on N/S faces
        const Real bAvS = 0.5 * (rsY(KFVM_D_DECL(i, j, k), 1, qIdx, ConsVar::magy) +
                                 rsY(KFVM_D_DECL(i, j, k), 0, qIdx, ConsVar::magy));
        const Real bAvN = 0.5 * (rsY(KFVM_D_DECL(i, j + 1, k), 0, qIdx, ConsVar::magy) +
                                 rsY(KFVM_D_DECL(i, j + 1, k), 1, qIdx, ConsVar::magy));
        divB += twt * (bAvN - bAvS) / geom.dy;
#if (SPACE_DIM == 3)
        // By normal on N/S faces
        const Real bAvB = 0.5 * (rsZ(i, j, k, 1, qIdx, ConsVar::magz) +
                                 rsZ(i, j, k, 0, qIdx, ConsVar::magz));
        const Real bAvT = 0.5 * (rsZ(i, j, k + 1, 0, qIdx, ConsVar::magz) +
                                 rsZ(i, j, k + 1, 1, qIdx, ConsVar::magz));
        divB += twt * (bAvT - bAvB) / geom.dz;
#endif
      }
    }

    // Find and fill forcing
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
          MHD_8W::SourceTerm(t, x, y, z, eosParams, userParams, state, force);

          // Pull out velocities and u.B term
          const Real vx = state[ConsVar::momx] / state[ConsVar::dens];
          const Real vy = state[ConsVar::momy] / state[ConsVar::dens];
          const Real vz = state[ConsVar::momz] / state[ConsVar::dens];
          const Real uDb = vx * state[ConsVar::magx] + vy * state[ConsVar::magy] +
                           vz * state[ConsVar::magz];

          // Fill source view user forcing and 8 wave terms
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::dens) =
              force[ConsVar::dens];

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::momx) =
              force[ConsVar::momx] - divB * state[ConsVar::magx];
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::momy) =
              force[ConsVar::momy] - divB * state[ConsVar::magy];
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::momz) =
              force[ConsVar::momz] - divB * state[ConsVar::magz];

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::etot) =
              force[ConsVar::etot] - divB * uDb;

          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::magx) =
              force[ConsVar::magx] - divB * vx;
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::magy) =
              force[ConsVar::magy] - divB * vy;
          source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), ConsVar::magz) =
              force[ConsVar::magz] - divB * vz;
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

} // namespace Physics

} // End namespace KFVM

#endif
