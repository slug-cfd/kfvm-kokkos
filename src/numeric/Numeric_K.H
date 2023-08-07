#ifndef NUMERIC_K_H_
#define NUMERIC_K_H_

// File: Numeric_K.H
// Purpose: Define compute kernels for various simple numerical methods
//          referencing data from Numeric.H

#include <cstdint>

#include <Kokkos_Core.hpp>

#include <Definitions.H>
#include <limits>

#include "../Dimension.H"
#include "../Geometry.H"
#include "../ParameterStructs.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "../physics/Physics_K.H"
#include "Numeric.H"

namespace KFVM {

namespace {
#include "InitialCondition.H"
}

namespace Numeric {

template <class DataViewType>
struct IntegrateIC_K {
  DataViewType v;
  QuadRuleView ab, wt;
  Geometry geom;
  EosParameters eosParams;
  UserParameters userParams;

  IntegrateIC_K(const DataViewType &v_, const QuadRuleView &ab_, const QuadRuleView &wt_,
                const Geometry &geom_, const EosParameters &eosParams_,
                const UserParameters &userParams_)
      : v(v_), ab(ab_), wt(wt_), geom(geom_), eosParams(eosParams_),
        userParams(userParams_) {}

  // 2D version
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Center of cell (i,j)
    Real xCtr = geom.xCoord(i), yCtr = geom.yCoord(j);

    // Zero out state
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      v(i, j, nV) = Real(0.0);
    }

    // Loop over quadrature points and accumulate integral
    for (idx_t nX = 0; nX < NUM_QUAD_PTS; nX++) {
      for (idx_t nY = 0; nY < NUM_QUAD_PTS; nY++) {
        // Quadrature point
        Real abX = sclAb(xCtr, geom.dx, ab(nX));
        Real abY = sclAb(yCtr, geom.dy, ab(nY));

        // IC at quadrature point
        Real state[NUM_VARS];
        InitialCondition(abX, abY, geom.zLo, eosParams, userParams, state);

        // Add each component into IC
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, nV) += wt(nX) * wt(nY) * state[nV];
        }
      }
    }
  }

  // 3D version
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k) const {
    // Center of cell (i,j,k)
    Real xCtr = geom.xCoord(i), yCtr = geom.yCoord(j), zCtr = geom.zCoord(k);

    // Zero out state
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      v(i, j, k, nV) = Real(0.0);
    }

    // Loop over quadrature points and accumulate integral
    for (idx_t nX = 0; nX < NUM_QUAD_PTS; nX++) {
      for (idx_t nY = 0; nY < NUM_QUAD_PTS; nY++) {
        for (idx_t nZ = 0; nZ < NUM_QUAD_PTS; nZ++) {
          // Quadrature point
          Real abX = sclAb(xCtr, geom.dx, ab(nX));
          Real abY = sclAb(yCtr, geom.dy, ab(nY));
          Real abZ = sclAb(zCtr, geom.dz, ab(nZ));

          // IC at quadrature point
          Real state[NUM_VARS];
          InitialCondition(abX, abY, abZ, eosParams, userParams, state);

          // Add each component into IC
          for (idx_t nV = 0; nV < NUM_VARS; nV++) {
            v(i, j, k, nV) += wt(nX) * wt(nY) * wt(nZ) * state[nV];
          }
        }
      }
    }
  }
};

template <class RhsViewType>
struct IntegrateRHS_K {
  RhsViewType rhs;
  FaceDataView KFVM_D_DECL(flX, flY, flZ);
  SourceDataView sources;
  bool haveSources;
  QuadRuleView ab, wt;
  Geometry geom;

  IntegrateRHS_K(const RhsViewType &rhs_,
                 KFVM_D_DECL(const FaceDataView &flX_, const FaceDataView &flY_,
                             const FaceDataView &flZ_),
                 const SourceDataView &sources_, bool haveSources_,
                 const QuadRuleView &ab_, const QuadRuleView &wt_, const Geometry &geom_)
      : rhs(rhs_), KFVM_D_DECL(flX(flX_), flY(flY_), flZ(flZ_)), sources(sources_),
        haveSources(haveSources_), ab(ab_), wt(wt_), geom(geom_) {}

#if (SPACE_DIM == 2)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Zero out RHS before accumulating fluxes and sources
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      rhs(i, j, nV) = Real(0.0);
    }

    // Integrate fluxes
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        // x-direction
        rhs(i, j, nV) +=
            wt(nQ) * (flX(i, j, 1, nQ, nV) - flX(i + 1, j, 0, nQ, nV)) / geom.dx;
        // y-direction
        rhs(i, j, nV) +=
            wt(nQ) * (flY(i, j, 1, nQ, nV) - flY(i, j + 1, 0, nQ, nV)) / geom.dy;
      }
    }

    if (haveSources) {
      // Integrate source terms
      for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (idx_t nV = 0; nV < NUM_VARS; nV++) {
            rhs(i, j, nV) += wt(nQ) * wt(nR) * sources(i, j, nQ, nR, nV);
          }
        }
      }
    }
  }
#else
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k) const {
    // Zero out RHS before accumulating fluxes and sources
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      rhs(i, j, k, nV) = Real(0.0);
    }

    // Integrate fluxes
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
          idx_t qIdx = nQ * NUM_QUAD_PTS + nR;
          Real qWt = wt(nQ) * wt(nR);
          rhs(i, j, k, nV) +=
              qWt * (flX(i, j, k, 0, qIdx, nV) - flX(i + 1, j, k, 1, qIdx, nV)) / geom.dx;
          rhs(i, j, k, nV) +=
              qWt * (flY(i, j, k, 0, qIdx, nV) - flY(i, j + 1, k, 1, qIdx, nV)) / geom.dy;
          rhs(i, j, k, nV) +=
              qWt * (flZ(i, j, k, 0, qIdx, nV) - flZ(i, j, k + 1, 1, qIdx, nV)) / geom.dz;
        }
      }
    }

    // Integrate source terms
    if (haveSources) {
      for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (idx_t nS = 0; nS < NUM_QUAD_PTS; nS++) {
            for (idx_t nV = 0; nV < NUM_VARS; nV++) {
              rhs(i, j, k, nV) +=
                  wt(nQ) * wt(nR) * wt(nS) * sources(i, j, k, nQ, nR, nS, nV);
            }
          }
        }
      }
    }
  }
#endif
};

template <class UViewType, class KViewType>
struct RKFSAL_StagePre_K {
  UViewType U, Uprev;
  KViewType Uhat, Utmp, K;
  Real betaDt, bhatDt;

  RKFSAL_StagePre_K(const UViewType &U_, const KViewType &Uhat_, const KViewType &Utmp_,
                    const UViewType &Uprev_, const KViewType &K_, Real betaDt_,
                    Real bhatDt_)
      : U(U_), Uprev(Uprev_), Uhat(Uhat_), Utmp(Utmp_), K(K_), betaDt(betaDt_),
        bhatDt(bhatDt_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    for (int nV = 0; nV < NUM_VARS; nV++) {
      U(KFVM_D_DECL(i, j, k), nV) =
          Uprev(KFVM_D_DECL(i, j, k), nV) + betaDt * K(KFVM_D_DECL(i, j, k), nV);
      Uhat(KFVM_D_DECL(i, j, k), nV) =
          Uprev(KFVM_D_DECL(i, j, k), nV) + bhatDt * K(KFVM_D_DECL(i, j, k), nV);
      Utmp(KFVM_D_DECL(i, j, k), nV) = Uprev(KFVM_D_DECL(i, j, k), nV);
    }
  }
};

template <class UViewType, class KViewType>
struct RKFSAL_Stage_K {
  UViewType U, Uprev;
  KViewType Uhat, Utmp, K;
  Real delta, gam1, gam2, gam3, betaDt, bhatDt;

  RKFSAL_Stage_K(const UViewType &U_, const KViewType &Uhat_, const KViewType &Utmp_,
                 const UViewType &Uprev_, const KViewType &K_, Real delta_, Real gam1_,
                 Real gam2_, Real gam3_, Real betaDt_, Real bhatDt_)
      : U(U_), Uprev(Uprev_), Uhat(Uhat_), Utmp(Utmp_), K(K_), delta(delta_), gam1(gam1_),
        gam2(gam2_), gam3(gam3_), betaDt(betaDt_), bhatDt(bhatDt_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    for (int nV = 0; nV < NUM_VARS; nV++) {
      Utmp(KFVM_D_DECL(i, j, k), nV) += delta * U(KFVM_D_DECL(i, j, k), nV);
      U(KFVM_D_DECL(i, j, k), nV) =
          gam1 * U(KFVM_D_DECL(i, j, k), nV) + gam2 * Utmp(KFVM_D_DECL(i, j, k), nV) +
          gam3 * Uprev(KFVM_D_DECL(i, j, k), nV) + betaDt * K(KFVM_D_DECL(i, j, k), nV);
      Uhat(KFVM_D_DECL(i, j, k), nV) += bhatDt * K(KFVM_D_DECL(i, j, k), nV);
    }
  }
};

template <class KViewType>
struct RKFSAL_StageLast_K {
  KViewType Uhat, K;
  Real bhatDt;

  RKFSAL_StageLast_K(const KViewType &Uhat_, const KViewType &K_, Real bhatDt_)
      : Uhat(Uhat_), K(K_), bhatDt(bhatDt_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    for (int nV = 0; nV < NUM_VARS; nV++) {
      Uhat(KFVM_D_DECL(i, j, k), nV) += bhatDt * K(KFVM_D_DECL(i, j, k), nV);
    }
  }
};

template <class UViewType, class KViewType>
struct RKFSAL_ErrEst_K {
  UViewType U;
  KViewType Uhat;
  Real atol, rtol;

  RKFSAL_ErrEst_K(const UViewType &U_, const KViewType &Uhat_, Real atol_, Real rtol_)
      : U(U_), Uhat(Uhat_), atol(atol_), rtol(rtol_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &err,
                  Real &posFlag) const {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::isnan;
    using Kokkos::pow;

    bool hasNan = false;
    Real state[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      Real u = U(KFVM_D_DECL(i, j, k), nV), uhat = Uhat(KFVM_D_DECL(i, j, k), nV);
      state[nV] = u;
      hasNan = hasNan || isnan(u);
      err += pow((u - uhat) / (atol + rtol * fmax(fabs(u), fabs(uhat))), 2);
    }
    Real pos = hasNan ? -Real(1.0) : Physics::EOS<eqType>::positivity(state);
    posFlag = posFlag > pos ? pos : posFlag;
  }
};

// KXRCF Indicator as defined in https://doi.org/10.1016/j.apnum.2003.11.002 and
// https://doi.org/10.1016/j.apnum.2020.03.013
template <class WViewType>
struct WenoSelect_K {
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  QuadRuleView wts;
  EosParameters eosParams;
  Real wThresh;
  WViewType W;

  WenoSelect_K(KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                           const FaceDataView &rsZ_),
               const QuadRuleView &wts_, const EosParameters &eosParams_,
               const Real wThresh_, const WViewType &W_)
      : KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), wts(wts_), eosParams(eosParams_),
        wThresh(wThresh_), W(W_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  uint32_t &nWeno) const {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;

    const int nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    Real numer = 0.0, minVal = std::numeric_limits<Real>::max();

    for (int nR = 0; nR < nQuadZ; nR++) {
      const Real wtR = SPACE_DIM == 2 ? 1.0 : wts(nR);
      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        const Real wt = wtR * wts(nQ);
        Real uIn[NUM_VARS], uOut[NUM_VARS]; // Inner and outer states
        Real wIn, wOut;                     // normal velocity prefactor and indicators

        // West face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsX(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Positive is into on West face
        vPF = Physics::EOS<eqType>::flowdirx(uIn) > 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;

        // East face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Negative is into on East face
        vPF = Physics::EOS<eqType>::flowdirx(uIn) < 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;

        // South face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsY(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Positive is into on South face
        vPF = Physics::EOS<eqType>::flowdiry(uIn) > 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;

        // North face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Negative is into on North face
        vPF = Physics::EOS<eqType>::flowdiry(uIn) < 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;

#if (SPACE_DIM == 3)
        // Bottom face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsZ(KFVM_D_DECL(i, j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Positive is into on Bottom face
        vPF = Physics::EOS<eqType>::flowdirz(uIn) > 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;

        // Top face
        for (int nV = 0; nV < NUM_VARS; nV++) {
          uIn[nV] = rsY(KFVM_D_DECL(i, j, k + 1), 0, nR * NUM_QUAD_PTS + nQ, nV);
          uOut[nV] = rsY(KFVM_D_DECL(i, j, k + 1), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }
        // Negative is into on Top face
        vPF = Physics::EOS<eqType>::flowdirz(uIn) < 0.0 ? 1.0 : 0.0;
        wIn = Physics::EOS<eqType>::weno(uIn, eosParams);
        wOut = Physics::EOS<eqType>::weno(uOut, eosParams);
        minVal = fmin(minVal, fabs(wIn));
        numer += vPF * wt * fabs(wOut - wIn);
        denom += vPF * wt;
#endif
      }
    }

    // Calculate change in weno indicator
    denom = minVal * fmax(denom, 1.e-8);
    const Real w = fabs(numer) / denom;

    // set flags based on indicator and last call to PP limiter
    if (w >= wThresh || W(KFVM_D_DECL(i, j, k), 1) < 1.0) {
      W(KFVM_D_DECL(i, j, k), 0) = 1.0;
      nWeno++;
    } else {
      W(KFVM_D_DECL(i, j, k), 0) = 0.0;
    }
  }
};

} // namespace Numeric

} // namespace KFVM

#endif
