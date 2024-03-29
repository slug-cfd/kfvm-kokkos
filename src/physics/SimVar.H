#ifndef SIMVAR_H_
#define SIMVAR_H_

#include "../Types.H"

namespace KFVM {

template <EquationType ET>
struct SimVar {};

template <>
struct SimVar<EquationType::Hydro> {
  struct ConsVar {
    enum var : int { dens = 0, momx, momy, momz, etot };
  };
  struct PrimVar {
    enum var : int { dens = 0, velx, vely, velz, pres };
  };
  struct AuxVar {
    enum var : int { velx = 0, vely, velz, eint, pres };
  };
  struct StatVar {
    enum var : int { ekin = 0, enst };
  };

  typedef Real ConsVec[5];
  typedef Real PrimVec[5];
  typedef Real StatVec[2];
};

template <>
struct SimVar<EquationType::NavierStokes> {
  // Ensure that consvar and primvar match hydro to re-use
  // Riemann solvers
  struct ConsVar {
    enum var : int { dens = 0, momx, momy, momz, etot };
  };
  struct PrimVar {
    enum var : int { dens = 0, velx, vely, velz, pres };
  };
  struct AuxVar {
    enum var : int { velx = 0, vely, velz, eint, pres };
  };
  struct StatVar {
    enum var : int { ekin = 0, enst };
  };

  typedef Real ConsVec[5];
  typedef Real PrimVec[5];
  typedef Real StatVec[2];
};

template <>
struct SimVar<EquationType::MHD_GLM> {
  struct ConsVar {
    enum var : int { dens = 0, momx, momy, momz, etot, magx, magy, magz, psi };
  };
  struct PrimVar {
    enum var : int { dens = 0, velx, vely, velz, prsg, magx, magy, magz, psi };
  };
  struct AuxVar {
    enum var : int { velx = 0, vely, velz, eint, prsg, prsb };
  };
  struct StatVar {
    enum var : int { ekin = 0, enst, heli };
  };

  typedef Real ConsVec[9];
  typedef Real PrimVec[9];
  typedef Real StatVec[3];
};

template <>
struct SimVar<EquationType::MHD_8W> {
  struct ConsVar {
    enum var : int { dens = 0, momx, momy, momz, etot, magx, magy, magz };
  };
  struct PrimVar {
    enum var : int { dens = 0, velx, vely, velz, prsg, magx, magy, magz };
  };
  struct AuxVar {
    enum var : int { velx = 0, vely, velz, eint, prsg, prsb };
  };
  struct StatVar {
    enum var : int { ekin = 0, enst, heli };
  };

  typedef Real ConsVec[8];
  typedef Real PrimVec[8];
  typedef Real StatVec[3];
};

template <>
struct SimVar<EquationType::SRHydro> {
  struct ConsVar {
    enum var : int { dens = 0, momx, momy, momz, etot };
  };
  struct PrimVar {
    enum var : int { rho = 0, velx, vely, velz, lorz, pres };
  };
  struct AuxVar {
    enum var : int { rho = 0, velx, vely, velz, lorz, pres };
  };
  struct StatVar {};

  typedef Real ConsVec[5];
  typedef Real PrimVec[6];
  typedef Real StatVec[1];
};

template <>
struct SimVar<EquationType::LinAdv> {
  struct ConsVar {
    enum var : int { u = 0 };
  };
  struct PrimVar {
    enum var : int { u = 0 };
  };
  struct AuxVar {
    enum var : int { u = 0 };
  };
  struct StatVar {};

  typedef Real ConsVec[1];
  typedef Real PrimVec[1];
  typedef Real StatVec[1];
};

namespace Physics {

// Array reduction type for integrating flow statistics
// Adapted from Kokkos documentation here:
// https://kokkos.github.io/kokkos-core-wiki/ProgrammingGuide/Custom-Reductions-Built-In-Reducers-with-Custom-Scalar-Types.html
template <int N>
struct ReducibleArray {
  Real data[N];

  KOKKOS_INLINE_FUNCTION // Default constructor - Initialize to 0's
  ReducibleArray() {
    for (int i = 0; i < N; i++) {
      data[i] = 0.0;
    }
  }

  KOKKOS_INLINE_FUNCTION // Copy Constructor
  ReducibleArray(const ReducibleArray &rhs) {
    for (int i = 0; i < N; i++) {
      data[i] = rhs.data[i];
    }
  }

  KOKKOS_INLINE_FUNCTION // add operator
      ReducibleArray &
      operator+=(const ReducibleArray &src) {
    for (int i = 0; i < N; i++) {
      data[i] += src.data[i];
    }
    return *this;
  }
};

typedef ReducibleArray<NUM_STATS> FlowStatsArray;

} // end namespace Physics

} // end namespace KFVM

namespace Kokkos { // reduction identity must be defined in Kokkos namespace
template <>
struct reduction_identity<KFVM::Physics::FlowStatsArray> {
  KOKKOS_FORCEINLINE_FUNCTION static KFVM::Physics::FlowStatsArray sum() {
    return KFVM::Physics::FlowStatsArray();
  }
};
} // namespace Kokkos

#endif
