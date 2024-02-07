#ifndef USERBCS_H_
#define USERBCS_H_
// This holds all user defined boundary conditions
// All of these functions must exist, regardless of BC type being used,
// however they can be left empty and will be ignored when using the
// built in ones

// There are two BC functions for each boundary, one to fill ghost cells
// prior to reconstrution, and one to fill face values priot to calling
// the Riemann solver

// Western boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellWest(const DataViewType &U, int rad, int nX, int j, int k,
                                     int nH, Real x, Real y, Real z, Real t,
                                     EosParameters eosParams, UserParameters userParams,
                                     Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::exp;
  using Kokkos::tanh;
  using ConsVar = typename KFVM::SimVar<EquationType::NavierStokes>::ConsVar;

  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nX;
  (void)j;
  (void)k;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;

  const Real om = userParams.omega;
  const Real gamm = eosParams.gamma - 1.0;
  const Real delta = -1.e-3 * exp(-0.5 * y * y) * (0.95 + 0.05 * cos(M_PI * z / 10.0)) *
                     (cos(om * t) + cos(0.5 * om * t - 0.028) +
                      cos(0.25 * om * t + 0.141) + cos(0.125 * om * t + 0.391));

  const Real rho0 = 1.0, rho = rho0 + 0.05 * delta;
  const Real u0 = (tanh(2.0 * y) + 3.0) / 8.0, u = u0 + delta;
  const Real v = 0.6 * delta;
  const Real w = SPACE_DIM == 2 ? 0.0 : 0.2 * delta;
  const Real p0 = 1.0 / eosParams.gamma, p = p0 + 0.2 * delta;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceWest(const DataViewType &U, int j, int k, int nQ, Real x,
                                     Real y, Real z, Real t, EosParameters eosParams,
                                     UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::exp;
  using Kokkos::tanh;
  using Physics::NavierStokes::SimVar;

  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)j;
  (void)k;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;

  const Real om = userParams.omega;
  const Real gamm = eosParams.gamma - 1.0;
  const Real delta = -1.e-3 * exp(-0.5 * y * y) * (0.95 + 0.05 * cos(M_PI * z / 10.0)) *
                     (cos(om * t) + cos(0.5 * om * t - 0.028) +
                      cos(0.25 * om * t + 0.141) + cos(0.125 * om * t + 0.391));

  const Real rho0 = 1.0, rho = rho0 + 0.05 * delta;
  const Real u0 = (tanh(2.0 * y) + 3.0) / 8.0, u = u0 + delta;
  const Real v = 0.6 * delta;
  const Real w = SPACE_DIM == 2 ? 0.0 : 0.2 * delta;
  const Real p0 = 1.0 / eosParams.gamma, p = p0 + 0.2 * delta;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;
  state[ConsVar::etot] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
}

// Eastern boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellEast(const DataViewType &U, int rad, int nX, int j, int k,
                                     int nH, Real x, Real y, Real z, Real t,
                                     EosParameters eosParams, UserParameters userParams,
                                     Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nX;
  (void)j;
  (void)k;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceEast(const DataViewType &U, int j, int k, int nQ, Real x,
                                     Real y, Real z, Real t, EosParameters eosParams,
                                     UserParameters userParams, Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)j;
  (void)k;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

// Southern boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellSouth(const DataViewType &U, int rad, int nY, int i,
                                      int k, int nH, Real x, Real y, Real z, Real t,
                                      EosParameters eosParams, UserParameters userParams,
                                      Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nY;
  (void)i;
  (void)k;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceSouth(const DataViewType &U, int i, int k, int nQ, Real x,
                                      Real y, Real z, Real t, EosParameters eosParams,
                                      UserParameters userParams, Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)i;
  (void)k;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

// Northern boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellNorth(const DataViewType &U, int rad, int nY, int i,
                                      int k, int nH, Real x, Real y, Real z, Real t,
                                      EosParameters eosParams, UserParameters userParams,
                                      Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nY;
  (void)i;
  (void)k;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceNorth(const DataViewType &U, int i, int k, int nQ, Real x,
                                      Real y, Real z, Real t, EosParameters eosParams,
                                      UserParameters userParams, Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)i;
  (void)k;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

// Bottom boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellBottom(const DataViewType &U, int rad, int nZ, int i,
                                       int j, int nH, Real x, Real y, Real z, Real t,
                                       EosParameters eosParams, UserParameters userParams,
                                       Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nZ;
  (void)i;
  (void)j;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void
FaceBottom(const DataViewType &U, int i, int j, int nQ, Real x, Real y, Real z, Real t,
           EosParameters eosParams, UserParameters userParams, Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)i;
  (void)j;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

// Top boundary
template <class DataViewType>
KOKKOS_INLINE_FUNCTION void CellTop(const DataViewType &U, int rad, int nZ, int i, int j,
                                    int nH, Real x, Real y, Real z, Real t,
                                    EosParameters eosParams, UserParameters userParams,
                                    Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)rad;
  (void)nZ;
  (void)i;
  (void)j;
  (void)nH;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceTop(const DataViewType &U, int i, int j, int nQ, Real x,
                                    Real y, Real z, Real t, EosParameters eosParams,
                                    UserParameters userParams, Real state[NUM_VARS]) {
  // Void casts to ignore unused parameter warnings
  (void)U;
  (void)i;
  (void)j;
  (void)nQ;
  (void)x;
  (void)y;
  (void)z;
  (void)t;
  (void)eosParams;
  (void)userParams;
  (void)state;
}

#endif
