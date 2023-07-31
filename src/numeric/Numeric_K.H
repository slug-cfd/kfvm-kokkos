#ifndef NUMERIC_K_H_
#define NUMERIC_K_H_

// File: Numeric_K.H
// Purpose: Define compute kernels for various simple numerical methods
//          referencing data from Numeric.H

#include <cstdint>
#include <cstdio>

#include <Kokkos_Core.hpp>

#include <Definitions.H>
#include <limits>

#include "../Geometry.H"
#include "../ParameterStructs.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "../physics/Physics_K.H"
#include "Dimension.H"
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
  Geometry<geomType> geom;
  EosParameters eosParams;
  UserParameters userParams;

  IntegrateIC_K(const DataViewType &v_, const QuadRuleView &ab_, const QuadRuleView &wt_,
                const Geometry<geomType> &geom_, const EosParameters &eosParams_,
                const UserParameters &userParams_)
      : v(v_), ab(ab_), wt(wt_), geom(geom_), eosParams(eosParams_),
        userParams(userParams_) {}

  // 2D version
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Zero out state
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      v(i, j, nV) = Real(0.0);
    }

    // Loop over quadrature points and accumulate integral
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Get physical space coordinate and Jacobian
        Vec4 xyzJ =
            geom.physCoordJac(KFVM_D_DECL(i, j, 0), KFVM_D_DECL(ab(nQ), ab(nR), 0.0));

        // IC at quadrature point
        Real state[NUM_VARS];
        InitialCondition(xyzJ.v1, xyzJ.v2, xyzJ.v3, eosParams, userParams, state);

        // Add each component into IC
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, nV) += xyzJ.v4 * wt(nQ) * wt(nR) * state[nV];
        }
      }
    }
  }

  // 3D version
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k) const {
    // Zero out state
    for (idx_t nV = 0; nV < NUM_VARS; nV++) {
      v(i, j, k, nV) = Real(0.0);
    }

    // Loop over quadrature points and accumulate integral
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        for (idx_t nS = 0; nS < NUM_QUAD_PTS; nS++) {
          // Get physical space coordinate and Jacobian
          Vec4 xyzJ = geom.physCoordJac(KFVM_D_DECL(i, j, k),
                                        KFVM_D_DECL(ab(nQ), ab(nR), ab(nS)));

          // IC at quadrature point
          Real state[NUM_VARS];
          InitialCondition(xyzJ.v1, xyzJ.v2, xyzJ.v3, eosParams, userParams, state);

          // Add each component into IC
          for (idx_t nV = 0; nV < NUM_VARS; nV++) {
            v(i, j, k, nV) += xyzJ.v4 * wt(nQ) * wt(nR) * wt(nS) * state[nV];
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
  Geometry<geomType> geom;

  IntegrateRHS_K(const RhsViewType &rhs_,
                 KFVM_D_DECL(const FaceDataView &flX_, const FaceDataView &flY_,
                             const FaceDataView &flZ_),
                 const SourceDataView &sources_, bool haveSources_,
                 const QuadRuleView &ab_, const QuadRuleView &wt_,
                 const Geometry<geomType> &geom_)
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
            wt(nQ) * (flX(i, j, 1, nQ, nV) - flX(i + 1, j, 0, nQ, nV)) / geom.dXi;
        // y-direction
        rhs(i, j, nV) +=
            wt(nQ) * (flY(i, j, 1, nQ, nV) - flY(i, j + 1, 0, nQ, nV)) / geom.dEta;
      }
    }

    if (haveSources) {
      // Integrate source terms
      for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
          Vec4 xyzJ =
              geom.physCoordJac(KFVM_D_DECL(i, j, 0), KFVM_D_DECL(ab(nQ), ab(nR), 0.0));
          for (idx_t nV = 0; nV < NUM_VARS; nV++) {
            rhs(i, j, nV) += xyzJ.v4 * wt(nQ) * wt(nR) * sources(i, j, nQ, nR, nV);
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
              qWt * (flX(i, j, k, 0, qIdx, nV) - flX(i + 1, j, k, 1, qIdx, nV)) /
              geom.dXi;
          rhs(i, j, k, nV) +=
              qWt * (flY(i, j, k, 0, qIdx, nV) - flY(i, j + 1, k, 1, qIdx, nV)) /
              geom.dEta;
          rhs(i, j, k, nV) +=
              qWt * (flZ(i, j, k, 0, qIdx, nV) - flZ(i, j, k + 1, 1, qIdx, nV)) /
              geom.dZeta;
        }
      }
    }

    // Integrate source terms
    if (haveSources) {
      for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (idx_t nS = 0; nS < NUM_QUAD_PTS; nS++) {
            Vec4 xyzJ = geom.physCoordJac(KFVM_D_DECL(i, j, k),
                                          KFVM_D_DECL(ab(nQ), ab(nR), ab(nS)));
            for (idx_t nV = 0; nV < NUM_VARS; nV++) {
              rhs(i, j, k, nV) +=
                  xyzJ.v4 * wt(nQ) * wt(nR) * wt(nS) * sources(i, j, k, nQ, nR, nS, nV);
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
  Geometry<geomType> geom;
  Real atol, rtol;

  RKFSAL_ErrEst_K(const UViewType &U_, const KViewType &Uhat_,
                  const Geometry<geomType> &geom_, Real atol_, Real rtol_)
      : U(U_), Uhat(Uhat_), geom(geom_), atol(atol_), rtol(rtol_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &err,
                  Real &posFlag) const {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::isnan;
    using Kokkos::pow;

    // Get mapped grid Jacobian
    const Real J = geom.jacobian(KFVM_D_DECL(i, j, k));

    bool hasNan = false;
    Real state[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      Real u = U(KFVM_D_DECL(i, j, k), nV), uhat = Uhat(KFVM_D_DECL(i, j, k), nV);
      state[nV] = u;
      hasNan = hasNan || isnan(u);
      err += pow((u - uhat) / (atol + rtol * fmax(fabs(u), fabs(uhat))), 2);
    }
    Real pos = hasNan ? -std::numeric_limits<Real>::max()
                      : Physics::EOS<eqType>::positivity(J, state);
    posFlag = posFlag > pos ? pos : posFlag;
  }
};

template <class UViewType, class WViewType>
struct RK_WenoSelect_K {
  UViewType U, Uprev;
  Geometry<geomType> geom;
  EosParameters eosParams;
  Real wThresh;
  WViewType W;

  RK_WenoSelect_K(const UViewType &U_, const UViewType &Uprev_,
                  const Geometry<geomType> &geom_, const EosParameters &eosParams_,
                  const Real wThresh_, const WViewType &W_)
      : U(U_), Uprev(Uprev_), geom(geom_), eosParams(eosParams_), wThresh(wThresh_),
        W(W_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  uint32_t &nWeno) const {
    using Kokkos::fabs;
    using Kokkos::fmax;
    using Kokkos::fmin;

    // Get mapped grid Jacobian
    const Real J = geom.jacobian(KFVM_D_DECL(i, j, k));

    // Extract states
    Real u0[NUM_VARS], u1[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      u0[nV] = Uprev(KFVM_D_DECL(i, j, k), nV);
      u1[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Calculate change in weno indicator
    Real w0 = Physics::EOS<eqType>::weno(eosParams, J, u0);
    Real w1 = Physics::EOS<eqType>::weno(eosParams, J, u1);
    Real w = fabs(w1 - w0) / fmin(w0, w1);

    // set flags based on indicator
    if (w < wThresh) {
      // weno not required, decrement counter
      W(KFVM_D_DECL(i, j, k), 0) = fmax(0.0, W(KFVM_D_DECL(i, j, k), 0) - 1.0);
    } else {
      // Set counter for full weno
      W(KFVM_D_DECL(i, j, k), 0) = Real(STENCIL_RADIUS + 1);
    }

    // Also set flags based on last call to PP limiter
    if (W(KFVM_D_DECL(i, j, k), 1) < 1.0 && W(KFVM_D_DECL(i, j, k), 0) < 2.0) {
      W(KFVM_D_DECL(i, j, k), 0) = 2.0;
    }

    // Increment counter if flag set
    if (W(KFVM_D_DECL(i, j, k), 0) > 0.0) {
      nWeno++;
    }
  }
};

} // namespace Numeric

} // namespace KFVM

#endif
