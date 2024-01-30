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
KOKKOS_INLINE_FUNCTION void FaceWest(const DataViewType &U, int j, int k, int nQ, Real x,
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
  using Kokkos::sqrt;
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

  const Real R = userParams.inletRadius;
  const Real vel = userParams.mach;
  const Real gam = eosParams.gamma, gamm = gam - 1.0;

  const Real rsq = SPACE_DIM == 2 ? x * x : x * x + z * z;

  // Inflow/outflow selector
  const Real tp = rsq <= R * R ? 1.0 : 0.0;

  // BC as mixture of inlet and outflow using taper
  const Real rhoOut = U(KFVM_D_DECL(i, rad, k), 0), rho = tp * gam + (1.0 - tp) * rhoOut;
  const Real uOut = U(KFVM_D_DECL(i, rad, k), 1) / rhoOut, u = (1.0 - tp) * uOut;
  const Real vOut = U(KFVM_D_DECL(i, rad, k), 2) / rhoOut,
             v = tp * vel + (1.0 - tp) * vOut;
  const Real wOut = U(KFVM_D_DECL(i, rad, k), 3) / rhoOut, w = (1.0 - tp) * wOut;
  const Real pOut = gamm * (U(KFVM_D_DECL(i, rad, k), 4) -
                            0.5 * rhoOut * (uOut * uOut + vOut * vOut + wOut * wOut));
  const Real p = tp + (1.0 - tp) * pOut;

  // Assign state
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = rho * w;
  state[4] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceSouth(const DataViewType &U, int i, int k, int nQ, Real x,
                                      Real y, Real z, Real t, EosParameters eosParams,
                                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;

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

  const Real R = userParams.inletRadius;
  const Real vel = userParams.mach;
  const Real gam = eosParams.gamma, gamm = gam - 1.0;

  const Real rsq = SPACE_DIM == 2 ? x * x : x * x + z * z;

  // Inflow/outflow selector
  const Real tp = rsq <= R * R ? 1.0 : 0.0;

  // BC as mixture of inlet and outflow using taper
  const Real rhoOut = U(KFVM_DM_DECL(i, k), 1, nQ, 0),
             rho = tp * gam + (1.0 - tp) * rhoOut;
  const Real uOut = U(KFVM_DM_DECL(i, k), 1, nQ, 1) / rhoOut, u = (1.0 - tp) * uOut;
  const Real vOut = U(KFVM_DM_DECL(i, k), 1, nQ, 2) / rhoOut,
             v = tp * vel + (1.0 - tp) * vOut;
  const Real wOut = U(KFVM_DM_DECL(i, k), 1, nQ, 3) / rhoOut, w = (1.0 - tp) * wOut;
  const Real pOut = gamm * (U(KFVM_DM_DECL(i, k), 1, nQ, 4) -
                            0.5 * rhoOut * (uOut * uOut + vOut * vOut + wOut * wOut));
  const Real p = tp + (1.0 - tp) * pOut;

  // Assign state
  state[0] = rho;
  state[1] = rho * u;
  state[2] = rho * v;
  state[3] = rho * w;
  state[4] = p / gamm + 0.5 * rho * (u * u + v * v + w * w);
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
