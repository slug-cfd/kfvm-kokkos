#ifndef PARAMETERSTRUCTS_H_
#define PARAMETERSTRUCTS_H_

// File: ParameterStructs.H
// Purpose: Hold all fluid properties in an easy to pass-by-value way

#include <cstddef>
#include <random>
#include <string>

#include <Definitions.H>

#include "Types.H"

namespace KFVM {

struct ParameterStructs {
  virtual bool set(const std::string &, const std::string &) = 0;
};

struct EosParameters : ParameterStructs {
  Real gamma;      // Ratio of specific heats
  Real Re, Pr;     // Reynolds, Prandtl numbers for N-S
  Real wenoThresh; // Threshold for weno selector
  Real ch_glm;     // Hyperbolic cleaning speed for GLM
  Real forceTOff;  // Cut off time for forcing
  Real forceRatio; // Compressive to solenoidal
  Real forceEDot;  // Energy injection rate
  Real forceStr;   // Forcing amplitude, calculated not set

  // Entries only needed if random forcing is enabled
#ifdef ENABLE_RANDOM_FORCING
  ForcingModeView fAmp;
#endif

  EosParameters()
      : gamma(1.4), Re(1.0), Pr(0.71), wenoThresh(10.0), ch_glm(0.0), forceRatio(1.0),
        forceEDot(2.0), forceStr(1.0)
#ifdef ENABLE_RANDOM_FORCING
        ,
        fAmp("ForcingModes")
#endif
  {
// fill forcing mode information if enabled
#ifdef ENABLE_RANDOM_FORCING
    auto h_fAmp = Kokkos::create_mirror(fAmp);
    Kokkos::deep_copy(h_fAmp, 0.0);
    int nm = 0;
    const int kMax = MAX_FORCING_MODE;
    const int kMaxSq = kMax * kMax, kMaxZ = SPACE_DIM == 2 ? 0 : kMax;
    for (int i = 0; i <= kMax; i++) {
      for (int j = -kMax; j <= kMax; j++) {
        for (int k = -kMaxZ; k <= kMaxZ; k++) {
          int ksq = i * i + j * j + k * k;
          if (ksq == 0 || ksq > kMaxSq) {
            // outside of band
            continue;
          }
          if (i == 0 && (j < 0 || k < 0)) {
            // don't duplicate wavevectors on axis boundaries
            continue;
          }
          // Fall through to allowed wavevector, add and increment counter
          h_fAmp(nm, 0, 0) = i;
          h_fAmp(nm, 1, 0) = j;
#if (SPACE_DIM == 3)
          h_fAmp(nm, 2, 0) = k;
#endif
          nm++;
        }
      }
    }
    Kokkos::deep_copy(fAmp, h_fAmp);
#endif
  }

  bool set(const std::string &, const std::string &);

  void updateForcing(const Real, int, int, const MPI_Datatype, MPI_Comm);
};

// Bring in user defined parameters
#include <UserParameters.H>

} // namespace KFVM

#endif
