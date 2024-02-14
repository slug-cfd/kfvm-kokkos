#ifndef NAVIERSTOKES_K_H_
#define NAVIERSTOKES_K_H_

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
#include "NavierStokes_EOS.H"
#include "physics/Physics_K.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::NavierStokes> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;

  const Real gamm;
  const Real rho, u, v, w, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const ConsVec ref_, Real gam_)
      : gamm(gam_ - Real(1.0)), rho(ref_[ConsVar::dens]), u(ref_[ConsVar::momx] / rho),
        v(ref_[ConsVar::momy] / rho), w(ref_[ConsVar::momz] / rho),
        vsq(Real(0.5) * (u * u + v * v + w * w)) {}

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
struct AuxVars<EquationType::NavierStokes, UViewType> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using AuxVar = typename SimVar<EquationType::NavierStokes>::AuxVar;
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;

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
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::NavierStokes>::eint(ua);
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) =
        EOS<EquationType::NavierStokes>::pres(eosParams.gamma, ua);
  }
};

template <class UViewType>
struct FlowStats<EquationType::NavierStokes, UViewType> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using StatVar = typename SimVar<EquationType::NavierStokes>::StatVar;

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

template <>
struct SpeedEstimate_K<EquationType::NavierStokes> {
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

template <class UViewType, class RViewType>
struct Viscosity_K<EquationType::NavierStokes, UViewType, RViewType> {
  using ConsVar = typename SimVar<EquationType::NavierStokes>::ConsVar;
  using PrimVar = typename SimVar<EquationType::NavierStokes>::PrimVar;
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;
  using PrimVec = typename SimVar<EquationType::NavierStokes>::PrimVec;

  UViewType U;
  RViewType RHS;
  Geometry geom;
  EosParameters eosParams;

  Viscosity_K(const UViewType &U_, const RViewType &RHS_, const Geometry geom_,
              const EosParameters eosParams_)
      : U(U_), RHS(RHS_), geom(geom_), eosParams(eosParams_) {}

#if (SPACE_DIM == 2)
  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    const Real rdx = 1.0 / geom.dx, rdy = 1.0 / geom.dy;
    const Real gamma = eosParams.gamma, Re = eosParams.Re, Pr = eosParams.Pr;

    Real u[3][3], v[3][3], w[3][3], T[3][3];
    for (int xo = -1; xo <= 1; xo++) {
      for (int yo = -1; yo <= 1; yo++) {
        ConsVec Uo;
        PrimVec Vo;
        for (int nV = 0; nV < NUM_VARS; nV++) {
          Uo[nV] = U(i + xo, j + yo, nV);
        }
        Physics::EOS<EquationType::NavierStokes>::cons2prim(gamma, Uo, Vo);
        u[xo + 1][yo + 1] = Vo[PrimVar::velx];
        v[xo + 1][yo + 1] = Vo[PrimVar::vely];
        w[xo + 1][yo + 1] = Vo[PrimVar::velz];
        T[xo + 1][yo + 1] = Physics::EOS<EquationType::NavierStokes>::temp(gamma, Uo);
      }
    }

    // Fluxes on x-faces
    Real px[4], mx[4];
    px[0] = 4.0 * rdx * (u[2][1] - u[1][1]) / 3.0;
    px[0] -= rdy * (v[1][2] + v[2][2] - v[1][0] - v[2][0]) / 6.0;

    px[1] = rdy * (u[1][2] + u[2][2] - u[1][0] - u[2][0]) / 4.0;
    px[1] += rdx * (v[2][1] - v[1][1]);

    px[2] = rdx * (w[2][1] - w[1][1]);

    mx[0] = 4.0 * rdx * (u[1][1] - u[0][1]) / 3.0;
    mx[0] -= rdy * (v[0][2] + v[1][2] - v[0][0] - v[1][0]) / 6.0;

    mx[1] = rdy * (u[0][2] + u[1][2] - u[0][0] - u[1][0]) / 4.0;
    mx[1] += rdx * (v[1][1] - v[0][1]);

    mx[2] = rdx * (w[1][1] - w[0][1]);

    // Fluxes on y-faces
    Real py[4], my[4];
    py[0] = rdx * (v[2][2] + v[2][1] - v[0][2] - v[0][1]) / 4.0;
    py[0] += rdy * (u[1][2] - u[1][1]);

    py[1] = 4.0 * rdy * (v[1][2] - v[1][1]) / 3.0;
    py[1] -= rdx * (u[2][2] + u[2][1] - u[0][2] - u[0][1]) / 6.0;

    py[2] = rdy * (w[1][2] - w[1][1]);

    my[0] = rdx * (v[2][1] + v[2][0] - v[0][1] - v[0][0]) / 4.0;
    my[0] += rdy * (u[1][1] - u[1][0]);

    my[1] = 4.0 * rdy * (v[1][1] - v[1][0]) / 3.0;
    my[1] -= rdx * (u[2][1] + u[2][0] - u[0][1] - u[0][0]) / 6.0;

    my[2] = rdy * (w[1][1] - w[1][0]);

    // Temperature fluxes
    px[3] = rdx * (T[2][1] - T[1][1]);
    mx[3] = rdx * (T[1][1] - T[0][1]);
    py[3] = rdy * (T[1][2] - T[1][1]);
    my[3] = rdy * (T[1][1] - T[1][0]);

    // Net momentum change
    const Real rRe = 1.0 / Re;
    const Real sru = rRe * (rdx * (px[0] - mx[0]) + rdy * (py[0] - my[0]));
    const Real srv = rRe * (rdx * (px[1] - mx[1]) + rdy * (py[1] - my[1]));
    const Real srw = rRe * (rdx * (px[2] - mx[2]) + rdy * (py[2] - my[2]));

    // Net temperature term
    const Real rAl = 1.0 / (Pr * Re);
    const Real ste = rAl * (rdx * (px[3] - mx[3]) + rdy * (py[3] - my[3]));

    // Add into RHS
    RHS(i, j, ConsVar::momx) += sru;
    RHS(i, j, ConsVar::momy) += srv;
    RHS(i, j, ConsVar::momz) += srw;
    RHS(i, j, ConsVar::etot) += ste + u[1][1] * sru + v[1][1] * srv + w[1][1] * srw;
  }
#else
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k) const {
    const Real rdx = 1.0 / geom.dx, rdy = 1.0 / geom.dy, rdz = 1.0 / geom.dz;
    const Real gamma = eosParams.gamma, Re = eosParams.Re, Pr = eosParams.Pr;

    Real u[3][3][3], v[3][3][3], w[3][3][3], T[3][3][3];
    for (int xo = -1; xo <= 1; xo++) {
      for (int yo = -1; yo <= 1; yo++) {
        for (int zo = -1; zo <= 1; zo++) {
          ConsVec Uo;
          PrimVec Vo;
          for (int nV = 0; nV < NUM_VARS; nV++) {
            Uo[nV] = U(i + xo, j + yo, k + zo, nV);
          }
          Physics::EOS<EquationType::NavierStokes>::cons2prim(gamma, Uo, Vo);
          u[xo + 1][yo + 1][zo + 1] = Vo[PrimVar::velx];
          v[xo + 1][yo + 1][zo + 1] = Vo[PrimVar::vely];
          w[xo + 1][yo + 1][zo + 1] = Vo[PrimVar::velz];
          T[xo + 1][yo + 1][zo + 1] =
              Physics::EOS<EquationType::NavierStokes>::temp(gamma, Uo);
        }
      }
    }

    // Fluxes on x-faces
    Real px[4], mx[4];
    px[0] = 4.0 * rdx * (u[2][1][1] - u[1][1][1]) / 3.0;
    px[0] -= rdy * (v[1][2][1] + v[2][2][1] - v[1][0][1] - v[2][0][1]) / 6.0;
    px[0] -= rdz * (w[1][1][2] + w[2][1][2] - w[1][1][0] - w[2][1][0]) / 6.0;

    px[1] = rdy * (u[1][2][1] + u[2][2][1] - u[1][0][1] - u[2][0][1]) / 4.0;
    px[1] += rdx * (v[2][1][1] - v[1][1][1]);

    px[2] = rdz * (u[2][1][2] + u[1][1][2] - u[2][1][0] - u[1][1][0]) / 4.0;
    px[2] += rdx * (w[2][1][1] - w[1][1][1]);

    mx[0] = 4.0 * rdx * (u[1][1][1] - u[0][1][1]) / 3.0;
    mx[0] -= rdy * (v[0][2][1] + v[1][2][1] - v[0][0][1] - v[1][0][1]) / 6.0;
    mx[0] -= rdz * (w[0][1][2] + w[1][1][2] - w[0][1][0] - w[1][1][0]) / 6.0;

    mx[1] = rdy * (u[0][2][1] + u[1][2][1] - u[0][0][1] - u[1][0][1]) / 4.0;
    mx[1] += rdx * (v[1][1][1] - v[0][1][1]);

    mx[2] = rdz * (u[1][1][2] + u[0][1][2] - u[1][1][0] - u[0][1][0]) / 4.0;
    mx[2] += rdx * (w[1][1][1] - w[0][1][1]);

    // Fluxes on y-faces
    Real py[4], my[4];
    py[0] = rdx * (v[2][2][1] + v[2][1][1] - v[0][2][1] - v[0][1][1]) / 4.0;
    py[0] += rdy * (u[1][2][1] - u[1][1][1]);

    py[1] = 4.0 * rdy * (v[1][2][1] - v[1][1][1]) / 3.0;
    py[1] -= rdx * (u[2][2][1] + u[2][1][1] - u[0][2][1] - u[0][1][1]) / 6.0;
    py[1] -= rdz * (w[1][2][2] + w[1][1][2] - w[1][2][0] - w[1][1][0]) / 6.0;

    py[2] = rdz * (v[1][2][2] + v[1][1][2] - v[1][2][0] - v[1][1][0]) / 4.0;
    py[2] += rdy * (w[1][2][1] - w[1][1][1]);

    my[0] = rdx * (v[2][1][1] + v[2][0][1] - v[0][1][1] - v[0][0][1]) / 4.0;
    my[0] += rdy * (u[1][1][1] - u[1][0][1]);

    my[1] = 4.0 * rdy * (v[1][1][1] - v[1][0][1]) / 3.0;
    my[1] -= rdx * (u[2][1][1] + u[2][0][1] - u[0][1][1] - u[0][0][1]) / 6.0;
    my[1] -= rdz * (w[1][1][2] + w[1][0][2] - w[1][1][0] - w[1][0][0]) / 6.0;

    my[2] = rdz * (v[1][1][2] + v[1][0][2] - v[1][1][0] - v[1][0][0]) / 4.0;
    my[2] += rdy * (w[1][1][1] - w[1][0][1]);

    // Fluxes on z-faces
    Real pz[4], mz[4];
    pz[0] = rdx * (w[2][2][1] + w[2][1][1] - w[0][2][1] - w[0][1][1]) / 4.0;
    pz[0] += rdz * (u[1][1][2] - u[1][1][1]);

    pz[1] = rdy * (w[1][2][2] + w[1][2][1] - w[1][0][2] - w[1][0][1]) / 4.0;
    pz[1] += rdz * (v[1][1][2] - v[1][1][1]);

    pz[2] = 4.0 * rdz * (w[1][1][2] - w[1][1][1]) / 3.0;
    pz[2] -= rdx * (u[2][2][1] + u[2][1][1] - u[0][2][1] - u[0][1][1]) / 6.0;
    pz[2] -= rdy * (v[1][2][2] + v[1][2][1] - v[1][0][2] - v[1][0][1]) / 6.0;

    mz[0] = rdx * (w[2][2][0] + w[2][1][0] - w[0][2][0] - w[0][1][0]) / 4.0;
    mz[0] += rdz * (u[1][1][1] - u[1][1][0]);

    mz[1] = rdy * (w[1][2][1] + w[1][2][0] - w[1][0][1] - w[1][0][0]) / 4.0;
    mz[1] += rdz * (v[1][1][1] - v[1][1][0]);

    mz[2] = 4.0 * rdz * (w[1][1][1] - w[1][1][0]) / 3.0;
    mz[2] -= rdx * (u[2][2][0] + u[2][1][0] - u[0][2][0] - u[0][1][0]) / 6.0;
    mz[2] -= rdy * (v[1][2][1] + v[1][2][0] - v[1][0][1] - v[1][0][0]) / 6.0;

    // Temperature fluxes
    px[3] = rdx * (T[2][1][1] - T[1][1][1]);
    mx[3] = rdx * (T[1][1][1] - T[0][1][1]);
    py[3] = rdy * (T[1][2][1] - T[1][1][1]);
    my[3] = rdy * (T[1][1][1] - T[1][0][1]);
    pz[3] = rdz * (T[1][1][2] - T[1][1][1]);
    mz[3] = rdz * (T[1][1][1] - T[1][1][0]);

    // Net momentum change
    const Real rRe = 1.0 / Re;
    const Real sru =
        rRe * (rdx * (px[0] - mx[0]) + rdy * (py[0] - my[0]) + rdz * (pz[0] - mz[0]));
    const Real srv =
        rRe * (rdx * (px[1] - mx[1]) + rdy * (py[1] - my[1]) + rdz * (pz[1] - mz[1]));
    const Real srw =
        rRe * (rdx * (px[2] - mx[2]) + rdy * (py[2] - my[2]) + rdz * (pz[2] - mz[2]));

    // Net temperature term
    const Real rAl = 1.0 / (Re * Pr);
    const Real ste =
        rAl * (rdx * (px[3] - mx[3]) + rdy * (py[3] - my[3]) + rdz * (pz[3] - mz[3]));

    // Add into RHS
    RHS(i, j, k, ConsVar::momx) += sru;
    RHS(i, j, k, ConsVar::momy) += srv;
    RHS(i, j, k, ConsVar::momz) += srw;
    RHS(i, j, k, ConsVar::etot) +=
        ste + u[1][1][1] * sru + v[1][1][1] * srv + w[1][1][1] * srw;
  }
#endif
};

namespace NavierStokes {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::NavierStokes, UViewType> {
  using ConsVec = typename SimVar<EquationType::NavierStokes>::ConsVec;

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
        wenoFlag(wenoFlag_), D(D_), ab(ab_), eosParams(eosParams_),
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
          NavierStokes::SourceTerm(t, x, y, z, eosParams, userParams, state, force);

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
