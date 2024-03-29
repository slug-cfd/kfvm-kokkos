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
  using ConsVar = typename KFVM::SimVar<EquationType::MHD_GLM>::ConsVar;
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

  const Real gam = eosParams.gamma, gamm = gam - 1.0;
  const Real R = userParams.inletRadius;
  const Real vy = userParams.mach;

  const Real rsq = SPACE_DIM == 2 ? x * x : x * x + z * z;

  // Always extrapolate B fields
  const Real bx = 0.0;                         // U(KFVM_D_DECL(i, rad, k), ConsVar::magx);
  const Real by = sqrt(2.0 / userParams.beta); // U(KFVM_D_DECL(i, rad, k), ConsVar::magy);
  const Real bz = 0.0;                         // U(KFVM_D_DECL(i, rad, k), ConsVar::magz);
  const Real bpsq = bx * bx + by * by + bz * bz;

  // Inflow conditions + B fields
  const Real inState[] = {
      gam, 0.0, gam * vy, 0.0, 1.0 / gamm + 0.5 * gam * vy * vy + 0.5 * bpsq,
      bx,  by,  bz,       0.0};

  if (rsq > R * R) { // outflow
    for (int nV = 0; nV < NUM_VARS; nV++) {
      state[nV] = U(KFVM_D_DECL(i, rad, k), nV);
    }
  } else { // Pure inflow
    for (int nV = 0; nV < NUM_VARS; nV++) {
      state[nV] = inState[nV];
    }
  }
}

template <class DataViewType>
KOKKOS_INLINE_FUNCTION void FaceSouth(const DataViewType &U, int i, int k, int nQ, Real x,
                                      Real y, Real z, Real t, EosParameters eosParams,
                                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename KFVM::SimVar<EquationType::MHD_GLM>::ConsVar;
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

  const Real gam = eosParams.gamma, gamm = gam - 1.0;
  const Real R = userParams.inletRadius;
  const Real vy = userParams.mach;

  const Real rsq = SPACE_DIM == 2 ? x * x : x * x + z * z;

  // Always extrapolate B fields
  const Real bx = 0.0;                         // U(KFVM_D_DECL(i, rad, k), ConsVar::magx);
  const Real by = sqrt(2.0 / userParams.beta); // U(KFVM_D_DECL(i, rad, k), ConsVar::magy);
  const Real bz = 0.0;                         // U(KFVM_D_DECL(i, rad, k), ConsVar::magz);
  const Real bpsq = bx * bx + by * by + bz * bz;

  // Inflow conditions + B fields
  const Real inState[] = {
      gam, 0.0, gam * vy, 0.0, 1.0 / gamm + 0.5 * gam * vy * vy + 0.5 * bpsq,
      bx,  by,  bz,       0.0};

  if (rsq > R * R) { // outflow
    for (int nV = 0; nV < NUM_VARS; nV++) {
      state[nV] = U(KFVM_DM_DECL(i, k), 1, nQ, nV);
    }
  } else { // Pure inflow
    for (int nV = 0; nV < NUM_VARS; nV++) {
      state[nV] = inState[nV];
    }
  }
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
