template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellWest(const DataViewType &U, idx_t rad, idx_t nX,
	      int j,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nX;
  (void) j; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellEast(const DataViewType &U, idx_t rad, idx_t nX,
	      int j,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nX;
  (void) j; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellSouth(const DataViewType &U, idx_t rad, idx_t nY,
	      int i,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real xRef = cos(8.0*M_PI*z)/12.0 + 1.0/4.0;
  const Real pf = x>xRef ? -1.0 : 1.0; // Only reflect on part of the domain

  state[SimVar::dens] = U(KFVM_D_DECL(i,rad + nH,k),SimVar::dens);
  state[SimVar::momx] = U(KFVM_D_DECL(i,rad + nH,k),SimVar::momx);
  state[SimVar::momy] = pf*U(KFVM_D_DECL(i,rad + nH,k),SimVar::momy);
  state[SimVar::momz] = U(KFVM_D_DECL(i,rad + nH,k),SimVar::momz);
  state[SimVar::etot] = U(KFVM_D_DECL(i,rad + nH,k),SimVar::etot);
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellNorth(const DataViewType &U, idx_t rad, idx_t nY,
	      int i,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real gam = 1.4,gamm = gam - 1.0;
  
  if(x < 8.66025*gam*t + 0.744017) {
    state[SimVar::dens] = 8.0;
    state[SimVar::momx] = 8.0*7.1447096;
    state[SimVar::momy] = -8.0*4.125;
    state[SimVar::momz] = 0.0;
    state[SimVar::etot] = 116.5/gamm + 4.0*(7.1447096*7.1447096 + 4.125*4.125);
  } else {
    state[SimVar::dens] = 1.4;
    state[SimVar::momx] = 0.0;
    state[SimVar::momy] = 0.0;
    state[SimVar::momz] = 0.0;
    state[SimVar::etot] = 1.0/gamm;
  }
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellBottom(const DataViewType &U, idx_t rad, idx_t nZ,
	      int i,int j,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nZ;
  (void) i; (void) j; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellTop(const DataViewType &U, idx_t rad, idx_t nZ,
	      int i,int j,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nZ;
  (void) i; (void) j; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}
