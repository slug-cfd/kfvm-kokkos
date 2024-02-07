#ifndef COMMON_K_H_
#define COMMON_K_H_

#include <cstdint>
#include <limits>
#include <math.h>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Dimension.H"
#include "../Geometry.H"
#include "../ParameterStructs.H"
#include "../Types.H"
#include "EquationTypes.H"
#include "SimVar.H"

namespace KFVM {

namespace Physics {

template <EquationType ET, class UViewType>
struct RandForcingIntegrate_K {
  using ConsVar = typename SimVar<ET>::ConsVar;

  UViewType U;
  EosParameters eosParams;
  Geometry geom;

  RandForcingIntegrate_K(const UViewType &U_, const EosParameters &eosParams_,
                         const Geometry geom_)
      : U(U_), eosParams(eosParams_), geom(geom_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k), Real &intA,
                  Real &intB) const {
    using Kokkos::cos;
    using Kokkos::sin;
#ifdef ENABLE_RANDOM_FORCING

    const Real str = 2.0; // Extra 2 to enforce conj. symm.
    const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

    // Sum forcing modes and track contributions
    Real fu = 0.0, fv = 0.0, fw = 0.0;
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
      const Real x = geom.xCoord(i), y = geom.yCoord(j);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0, z = geom.zCoord(0);
#else
      const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
#endif

      // Set k.x and set forcing from this mode ignoring projection
      const Real kdx = k1 * x + k2 * y + k3 * z;
      const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 2)
      const Real fwn = 0.0;
#else
      const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif

      // Project and add into total forcing
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

      fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
                   mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

      fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
                   mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

      fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
                   mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
    }

    intA += 0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw * fw);
    intB += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
            U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
            U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;
#endif
  }
};

template <EquationType ET, class UViewType>
struct RandForcingApply_K {
  using ConsVar = typename SimVar<ET>::ConsVar;

  UViewType U;
  EosParameters eosParams;
  Geometry geom;

  RandForcingApply_K(const UViewType &U_, const EosParameters &eosParams_,
                     const Geometry &geom_)
      : U(U_), eosParams(eosParams_), geom(geom_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    using Kokkos::cos;
    using Kokkos::sin;
#ifdef ENABLE_RANDOM_FORCING

    const Real str = 2.0 * eosParams.forceStr; // Extra 2 to enforce conj. symm.
    const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;

    // Sum forcing modes and track contributions
    Real fu = 0.0, fv = 0.0, fw = 0.0;
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
      const Real x = geom.xCoord(i), y = geom.yCoord(j);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0, z = geom.zCoord(0);
#else
      const Real k3 = eosParams.fAmp(n, 2, 0), z = geom.zCoord(k);
#endif

      // Set k.x and set forcing from this mode ignoring projection
      const Real kdx = k1 * x + k2 * y + k3 * z;
      const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 2)
      const Real fwn = 0.0;
#else
      const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif

      // Project and add into total forcing
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

      fu += str * ((fr + mfr * iksq * k1 * k1) * fun +
                   mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn));

      fv += str * ((fr + mfr * iksq * k2 * k2) * fvn +
                   mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn));

      fw += str * ((fr + mfr * iksq * k3 * k3) * fwn +
                   mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn));
    }

    // Energy in the perturbation as from above
    Real ePert =
        0.5 * U(KFVM_D_DECL(i, j, k), ConsVar::dens) * (fu * fu + fv * fv + fw * fw);
    ePert += U(KFVM_D_DECL(i, j, k), ConsVar::momx) * fu +
             U(KFVM_D_DECL(i, j, k), ConsVar::momy) * fv +
             U(KFVM_D_DECL(i, j, k), ConsVar::momz) * fw;

    // Update momenta and total energy
    U(KFVM_D_DECL(i, j, k), ConsVar::momx) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fu;
    U(KFVM_D_DECL(i, j, k), ConsVar::momy) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fv;
    U(KFVM_D_DECL(i, j, k), ConsVar::momz) += U(KFVM_D_DECL(i, j, k), ConsVar::dens) * fw;
    U(KFVM_D_DECL(i, j, k), ConsVar::etot) += ePert;
#endif
  }
};

} // namespace Physics

} // End namespace KFVM

#endif
