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
#include "../hydro/Hydro_Riemann.H"
#include "NavierStokes_EOS.H"
#include "physics/Physics_K.H"

namespace KFVM {

namespace Physics {

template <>
class VarConvert<EquationType::NavierStokes> {
  const Real gamm;
  const Real rho, u, v, w, vsq;

 public:
  KOKKOS_INLINE_FUNCTION
  VarConvert(const NavierStokes::ConsVec ref_, Real gam_)
      : gamm(gam_ - Real(1.0)), rho(ref_[NavierStokes::SimVar::dens]),
        u(ref_[NavierStokes::SimVar::momx] / rho),
        v(ref_[NavierStokes::SimVar::momy] / rho),
        w(ref_[NavierStokes::SimVar::momz] / rho),
        vsq(Real(0.5) * (u * u + v * v + w * w)) {}

  template <class T>
  KOKKOS_INLINE_FUNCTION void cons2LinPrim(T &state) {
    using NavierStokes::SimVar;

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
    using NavierStokes::SimVar;

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
struct AuxVars<EquationType::NavierStokes, UViewType> {
  UViewType U;
  AuxDataView V;
  EosParameters eosParams;

  AuxVars(const UViewType &U_, const AuxDataView V_, const EosParameters &eosParams_)
      : U(U_), V(V_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using NavierStokes::AuxVar;
    using NavierStokes::ConsVec;
    using NavierStokes::SimVar;

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
    V(KFVM_D_DECL(i, j, k), AuxVar::eint) = EOS<EquationType::NavierStokes>::eint(ua);
    V(KFVM_D_DECL(i, j, k), AuxVar::pres) =
        EOS<EquationType::NavierStokes>::pres(eosParams.gamma, ua);
  }
};

template <class UViewType>
struct FlowStats<EquationType::NavierStokes, UViewType> {
  UViewType U;
  Geometry geom;
  EosParameters eosParams;

  FlowStats(const UViewType &U_, const Geometry &geom_, const EosParameters &eosParams_)
      : U(U_), geom(geom_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  FlowStatsArray &stat) const {
    using NavierStokes::SimVar;
    using NavierStokes::StatVar;

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
struct RiemannSolverX_K<EquationType::NavierStokes, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverX_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using NavierStokes::ConsVec;
    using NavierStokes::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uR, vR);

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
struct RiemannSolverY_K<EquationType::NavierStokes, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverY_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                  Real &maxVel) const {
    using NavierStokes::ConsVec;
    using NavierStokes::PrimVec;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uR, vR);

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
struct RiemannSolverZ_K<EquationType::NavierStokes, rst> {
  FaceDataView U;
  EosParameters eosParams;

  RiemannSolverZ_K(const FaceDataView &U_, const EosParameters &eosParams_)
      : U(U_), eosParams(eosParams_) {}

#if (SPACE_DIM == 3)
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k, Real &maxVel) const {
    using NavierStokes::ConsVec;
    using NavierStokes::PrimVec;

    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // pull out states and convert to primitive
      ConsVec uL, uR, F;
      PrimVec vL, vR;
      for (int nV = 0; nV < NUM_VARS; nV++) {
        uL[nV] = U(KFVM_D_DECL(i, j, k), 0, nQ, nV);
        uR[nV] = U(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uL, vL);
      EOS<EquationType::NavierStokes>::cons2prim(eosParams.gamma, uR, vR);

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
struct SpeedEstimate_K<EquationType::NavierStokes> {
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

template <class UViewType, class RViewType>
struct Viscosity_K<EquationType::NavierStokes, UViewType, RViewType> {
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
    using Physics::NavierStokes::ConsVec;
    using Physics::NavierStokes::PrimVar;
    using Physics::NavierStokes::PrimVec;
    using Physics::NavierStokes::SimVar;

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
        T[xo + 1][yo + 1] = Vo[PrimVar::temp];
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
    RHS(i, j, SimVar::momx) += sru;
    RHS(i, j, SimVar::momy) += srv;
    RHS(i, j, SimVar::momz) += srw;
    RHS(i, j, SimVar::etot) += ste + u[1][1] * sru + v[1][1] * srv + w[1][1] * srw;
  }
#else
  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j, const idx_t k) const {
    using Physics::NavierStokes::ConsVec;
    using Physics::NavierStokes::PrimVar;
    using Physics::NavierStokes::PrimVec;
    using Physics::NavierStokes::SimVar;

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
          T[xo + 1][yo + 1][zo + 1] = Vo[PrimVar::temp];
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
    RHS(i, j, k, SimVar::momx) += sru;
    RHS(i, j, k, SimVar::momy) += srv;
    RHS(i, j, k, SimVar::momz) += srw;
    RHS(i, j, k, SimVar::etot) +=
        ste + u[1][1][1] * sru + v[1][1][1] * srv + w[1][1][1] * srw;
  }
#endif
};

namespace NavierStokes {
#include "SourceTerms.H"
}

template <class UViewType>
struct SourceTerms_K<EquationType::NavierStokes, UViewType> {
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
          NavierStokes::ConsVec state, force;
          // Real state[10],force[10];
          // Extract state and clear force vector
          for (int nV = 0; nV < NUM_VARS; nV++) {
            state[nV] = source(KFVM_D_DECL(i, j, k), KFVM_D_DECL(nQ, nR, nS), nV);
            force[nV] = 0;
          }

          // Fill force with user defined terms
          NavierStokes::SourceTerm(t, x, y, z, state, force);

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
struct PositivityPreserve_K<EquationType::NavierStokes, UDataView> {
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
    using NavierStokes::ConsVec;
    using NavierStokes::SimVar;

    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;

    // Save reference state
    ConsVec refState;
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }
    Real rDens = refState[SimVar::dens];

    // Find ranges of density, internal energy, and soundspeed
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
          Real eint = EOS<EquationType::NavierStokes>::eint(state);
          Real sdsp = EOS<EquationType::NavierStokes>::soundspeed(eosParams.gamma, state);
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
    Real kappa1 = Real(0.3), kappa2 = Real(0.3);
    Real eta =
        fmin(Real(1.0), fmax(Real(0.0), -(kappa1 * minSdSp + divV) / (kappa1 * minSdSp)));

    // Extend density and pressure ranges
    maxDens *= (Real(1.0) + kappa2 - kappa2 * eta);
    minDens *= (Real(1.0) - kappa2 + kappa2 * eta);
    minEint *= (Real(1.0) - kappa2 + kappa2 * eta);

    Real rDensDiffP = maxDens - rDens, rDensDiffM = rDens - minDens;

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

    // Enforce positive internal energy
    // Find mixing amount
    theta = Real(1.0);
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      ConsVec state;
      Real ei;
      // West face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
      // East face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsX(KFVM_D_DECL(i + 1, j, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
      // South face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
      // North face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsY(KFVM_D_DECL(i, j + 1, k), 0, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
#if (SPACE_DIM == 3)
      // Bottom face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k), 1, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
      // Top face
      for (int nV = 0; nV < NUM_VARS; nV++) {
        state[nV] = rsZ(KFVM_D_DECL(i, j, k + 1), 0, nQ, nV);
      }
      ei = EOS<EquationType::NavierStokes>::eint(state);
      theta = ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
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
            Real ei = EOS<EquationType::NavierStokes>::eint(state);
            theta =
                ei >= minEint ? theta : fmin(theta, nsEintPos(minEint, refState, state));
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
  Real nsEintPos(const Real emin, const NavierStokes::ConsVec refState,
                 const NavierStokes::ConsVec state) const {
    using Kokkos::fabs;
    using Kokkos::fmin;
    using NavierStokes::ConsVec;
    using NavierStokes::SimVar;

    Real a = 0.0, b = 0.1, c, dea, dec;
    ConsVec stTheta;
    dea = EOS<EquationType::NavierStokes>::eint(refState) - emin;

    // Scan b forward until sign changes
    for (int nIter = 0; nIter < 9; nIter++) {
      for (int nV = 0; nV < NUM_VARS; nV++) {
        stTheta[nV] = refState[nV] + b * (state[nV] - refState[nV]);
      }

      dec = EOS<EquationType::NavierStokes>::eint(stTheta) - emin;

      if (dea * dec < 0.0) {
        break;
      } else {
        // slide interval forward
        a += 0.1;
        b += 0.1;
        dea = dec;
      }
    }

    // Now do bisection since root is bracketed
    for (int nIter = 0; nIter < 10; nIter++) {
      // Midpoint and states
      c = Real(0.5) * (a + b);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        stTheta[nV] = refState[nV] + c * (state[nV] - refState[nV]);
      }

      // Energy differences at states a and c
      dec = EOS<EquationType::NavierStokes>::eint(stTheta) - emin;

      if (dec * dea < 0.0) {
        b = c;
      } else {
        a = c;
        dea = dec;
      }
    }

    // Round down to lower bound to guarantee a valid state
    return a;
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
