// This holds all user defined boundary conditions
// All of these functions must exist, regardless of BC type being used,
// however they can be left empty and will be ignored when using the
// built in ones

// There are two BC functions for each boundary, one to fill ghost cells
// prior to reconstrution, and one to fill face values priot to calling
// the Riemann solver


// Western boundary
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
void FaceWest(const DataViewType &U,idx_t nX,
	      int j,int k,int nQ,int nR,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nX;
  (void) j; (void) k; (void) nQ; (void) nR;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Eastern boundary
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
void FaceEast(const DataViewType &U,idx_t nX,
	      int j,int k,int nQ,int nR,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nX;
  (void) j; (void) k; (void) nQ; (void) nR;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Southern boundary
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

  const Real xRef = cos(2.0*M_PI*z)/12.0 + 1.0/4.0;
  const Real pf = x>xRef ? -1.0 : 1.0; // Only reflect on part of the domain

  state[SimVar::dens] =    U(KFVM_D_DECL(i,rad + nH,k),SimVar::dens);
  state[SimVar::momx] =    U(KFVM_D_DECL(i,rad + nH,k),SimVar::momx);
  state[SimVar::momy] = pf*U(KFVM_D_DECL(i,rad + nH,k),SimVar::momy);
  state[SimVar::momz] =    U(KFVM_D_DECL(i,rad + nH,k),SimVar::momz);
  state[SimVar::etot] =    U(KFVM_D_DECL(i,rad + nH,k),SimVar::etot);
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceSouth(const DataViewType &U,idx_t nY,
	       int i,int k,int nQ,int nR,
	       Real x,Real y,Real z,Real t,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nY;
  (void) i; (void) k; (void) nQ; (void) nR;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real xRef = cos(2.0*M_PI*z)/12.0 + 1.0/4.0;
  const Real pf = x>xRef ? -1.0 : 1.0; // Only reflect on part of the domain

  state[SimVar::dens] =    U(KFVM_D_DECL(i,1,k),FaceLabel::south,KFVM_DM_DECL(nQ,nR),SimVar::dens);
  state[SimVar::momx] =    U(KFVM_D_DECL(i,1,k),FaceLabel::south,KFVM_DM_DECL(nQ,nR),SimVar::momx);
  state[SimVar::momy] = pf*U(KFVM_D_DECL(i,1,k),FaceLabel::south,KFVM_DM_DECL(nQ,nR),SimVar::momy);
  state[SimVar::momz] =    U(KFVM_D_DECL(i,1,k),FaceLabel::south,KFVM_DM_DECL(nQ,nR),SimVar::momz);
  state[SimVar::etot] =    U(KFVM_D_DECL(i,1,k),FaceLabel::south,KFVM_DM_DECL(nQ,nR),SimVar::etot);
}

// Northern boundary
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
void FaceNorth(const DataViewType &U,idx_t nY,
	       int i,int k,int nQ,int nR,
	       Real x,Real y,Real z,Real t,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nY;
  (void) i; (void) k; (void) nQ; (void) nR;
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

// Bottom boundary
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
void FaceBottom(const DataViewType &U,idx_t nZ,
		int i,int j,int nQ,int nR,
		Real x,Real y,Real z,Real t,
		Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nZ;
  (void) i; (void) j; (void) nQ; (void) nR;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Top boundary
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

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceTop(const DataViewType &U,idx_t nZ,
	     int i,int j,int nQ,int nR,
	     Real x,Real y,Real z,Real t,
	     Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) nZ;
  (void) i; (void) j; (void) nQ; (void) nR;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}
