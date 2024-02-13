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
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;

  const Real gamm;
  const Real rho, u, v, w, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const ConsVec ref_, Real gam_)
      : gamm(gam_ - 1.0), rho(ref_[ConsVar::dens]), u(ref_[ConsVar::momx] / rho),
        v(ref_[ConsVar::momy] / rho), w(ref_[ConsVar::momz] / rho),
        vsq(0.5 * (u * u + v * v + w * w)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {

    // Doing etot first allows state to be overwritten safely
    state(ConsVar::etot) = gamm * (vsq * state(ConsVar::dens) - u * state(ConsVar::momx) -
                                   v * state(ConsVar::momy) - w * state(ConsVar::momz) +
                                   state(ConsVar::etot));

    state(ConsVar::momx) = (state(ConsVar::momx) - u * state(ConsVar::dens)) / rho;
    state(ConsVar::momy) = (state(ConsVar::momy) - v * state(ConsVar::dens)) / rho;
    state(ConsVar::momz) = (state(ConsVar::momz) - w * state(ConsVar::dens)) / rho;
  }

  template <class T>
  KOKKOS_INLINE_FUNCTION void linPrim2Cons(T &state) {

    // Doing etot first allows state to be overwritten safely
    state(ConsVar::etot) = (vsq * state(ConsVar::dens) + rho * u * state(ConsVar::momx) +
                            rho * v * state(ConsVar::momy) +
                            rho * w * state(ConsVar::momz) + state(ConsVar::etot) / gamm);

    state(ConsVar::momx) = (rho * state(ConsVar::momx) + u * state(ConsVar::dens));
    state(ConsVar::momy) = (rho * state(ConsVar::momy) + v * state(ConsVar::dens));
    state(ConsVar::momz) = (rho * state(ConsVar::momz) + w * state(ConsVar::dens));
  }
};

template <class UViewType>
struct AuxVars<EquationType::Hydro, UViewType> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using AuxVar = typename SimVar<EquationType::Hydro>::AuxVar;
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;

  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Try not to assume the order of SimVars
    ConsVec ua;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      ua[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Set velocities
    V(KFVM_D_DECL(i, j, k), AuxVar::velx) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momx) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::vely) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momy) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);
    V(KFVM_D_DECL(i, j, k), AuxVar::velz) =
        U(KFVM_D_DECL(i, j, k), ConsVar::momz) / U(KFVM_D_DECL(i, j, k), ConsVar::dens);

    // Internal energy and pressure
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::Hydro>::eint(ua);
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) =
        EOS<EquationType::Hydro>::pres(eosParams.gamma, ua);
  }
};

template <class UViewType>
struct FlowStats<EquationType::Hydro, UViewType> {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;
  using StatVar = typename SimVar<EquationType::Hydro>::StatVar;

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

    // Add into array for reduction
    const Real ifac = geom.dvol / geom.volume;
    stat.data[StatVar::ekin] += ifac * ek;
    stat.data[StatVar::enst] += ifac * et;
  }
};

template <RSType rst>
struct RiemannSolverX_K<EquationType::Hydro, rst> {
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

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
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

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
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;
  using PrimVec = typename SimVar<EquationType::Hydro>::PrimVec;

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
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &maxLam,
                  Real &maxVel) const {}
};

// template <class UViewType>
// struct RandForcingIntegrate_K<EquationType::Hydro, UViewType> {
//   using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

//   UViewType U;
//   EosParameters eosParams;
//   Geometry geom;

//   RandForcingIntegrate_K(const UViewType &U_, const EosParameters &eosParams_,
//                          const Geometry geom_)
//       : U(U_), eosParams(eosParams_), geom(geom_) {}

//   KOKKOS_INLINE_FUNCTION
//   void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &intA,
//                   Real &intB) const {
//     using Kokkos::cos;
//     using Kokkos::sin;
// #ifdef ENABLE_RANDOM_FORCING

//     const Real str = 2.0; // Extra 2 to enforce conj. symm.
//     const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

//     // Sum forcing modes and track contributions
//     Real fu = 0.0, fv = 0.0, fw = 0.0;
//     for (int n = 0; n < NUM_FORCING_MODES; n++) {
//       const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
//       const Real x = geom.xCoord(i), y = geom.yCoord(j);
// #if (SPACE_DIM == 2)
//       const Real k3 = 0.0, z = geom.zCoord(0);
// #else
//       const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
// #endif

//       // Set k.x and set forcing from this mode ignoring projection
//       const Real kdx = k1 * x + k2 * y + k3 * z;
//       const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
//       const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
// #if (SPACE_DIM == 2)
//       const Real fwn = 0.0;
// #else
//       const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
// #endif

//       // Project and add into total forcing
//       const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

//       fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
//                    mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

//       fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
//                    mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

//       fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
//                    mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
//     }

//     intA += 0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw *
//     fw); intB += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
//             U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
//             U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;
// #endif
//   }
// };

// template <class UViewType>
// struct RandForcingApply_K<EquationType::Hydro, UViewType> {
//   using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

//   UViewType U;
//   EosParameters eosParams;
//   Geometry geom;

//   RandForcingApply_K(const UViewType &U_, const EosParameters &eosParams_,
//                      const Geometry &geom_)
//       : U(U_), eosParams(eosParams_), geom(geom_) {}

//   KOKKOS_INLINE_FUNCTION
//   void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
//     using Kokkos::cos;
//     using Kokkos::sin;
// #ifdef ENABLE_RANDOM_FORCING

//     const Real str = 2.0 * eosParams.forceStr; // Extra 2 to enforce conj. symm.
//     const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

//     // Sum forcing modes and track contributions
//     Real fu = 0.0, fv = 0.0, fw = 0.0;
//     for (int n = 0; n < NUM_FORCING_MODES; n++) {
//       const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
//       const Real x = geom.xCoord(i), y = geom.yCoord(j);
// #if (SPACE_DIM == 2)
//       const Real k3 = 0.0, z = geom.zCoord(0);
// #else
//       const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
// #endif

//       // Set k.x and set forcing from this mode ignoring projection
//       const Real kdx = k1 * x + k2 * y + k3 * z;
//       const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
//       const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
// #if (SPACE_DIM == 2)
//       const Real fwn = 0.0;
// #else
//       const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
//                        eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
// #endif

//       // Project and add into total forcing
//       const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

//       fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
//                    mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

//       fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
//                    mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

//       fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
//                    mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
//     }

//     // Energy in the perturbation as from above
//     Real ePert =
//         0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw * fw);
//     ePert += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
//              U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
//              U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;

//     // Update momenta and total energy
//     U(KFVM_D_DECL(i, j, k), ConsVar::momx) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) *
//     fu; U(KFVM_D_DECL(i, j, k), ConsVar::momy) += U(KFVM_D_DECL(i, j, k),
//     ConsVar::dens) * fv; U(KFVM_D_DECL(i, j, k), ConsVar::momz) += U(KFVM_D_DECL(i, j,
//     k), ConsVar::dens) * fw; U(KFVM_D_DECL(i, j, k), ConsVar::etot) += ePert;
// #endif
//   }
// };

namespace Hydro {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::Hydro, UViewType> {
  using ConsVec = typename SimVar<EquationType::Hydro>::ConsVec;

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

  SourceTerms_K(const SourceDataView &source_,
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
          // Real state[10],force[10];
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            force[nV] = 0;
          }

          // Fill force with user defined terms
          Hydro::SourceTerm(t, x, y, z, eosParams, userParams, state, force);

          // overwrite source with forcing
          for (int nV = 0; nV < NUM_VARS; nV++) {
            source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV) = force[nV];
          }
        }
      }
    }
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
