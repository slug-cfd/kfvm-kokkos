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

  if (y * y + z * z < 1.0) { // inflow
    const Real gam = 5.0 / 3.0, gamm = gam - 1.0;
    const Real rho = 0.01, p = 0.01, vx = 0.99;
    const Real W = 1.0 / Kokkos::sqrt(1.0 - vx * vx), h = 1.0 + p * gam / (rho * gamm);
    state[0] = rho * W;
    state[1] = rho * W * W * h * vx;
    state[2] = 0.0;
    state[3] = 0.0;
    state[4] = rho * W * (h * W - 1.0) - p;
  } else { // outflow
    state[0] = U(KFVM_D_DECL(rad + nH, j, k), 0);
    state[1] = U(KFVM_D_DECL(rad + nH, j, k), 1);
    state[2] = U(KFVM_D_DECL(rad + nH, j, k), 2);
    state[3] = U(KFVM_D_DECL(rad + nH, j, k), 3);
    state[4] = U(KFVM_D_DECL(rad + nH, j, k), 4);
  }
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

  if (y * y + z * z < 1.0) { // inflow
    const Real gam = 5.0 / 3.0, gamm = gam - 1.0;
    const Real rho = 0.01, p = 0.01, vx = 0.99;
    const Real W = 1.0 / Kokkos::sqrt(1.0 - vx * vx), h = 1.0 + p * gam / (rho * gamm);
    state[0] = rho * W;
    state[1] = rho * W * W * h * vx;
    state[2] = 0.0;
    state[3] = 0.0;
    state[4] = rho * W * (h * W - 1.0) - p;
  } else { // outflow
    state[0] = U(KFVM_DM_DECL(j, k), 1, nQ, 0);
    state[1] = U(KFVM_DM_DECL(j, k), 1, nQ, 1);
    state[2] = U(KFVM_DM_DECL(j, k), 1, nQ, 2);
    state[3] = U(KFVM_DM_DECL(j, k), 1, nQ, 3);
    state[4] = U(KFVM_DM_DECL(j, k), 1, nQ, 4);
  }
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
