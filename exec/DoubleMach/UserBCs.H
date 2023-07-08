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
void CellWest(const DataViewType& U,int rad,int nX,
	      int j,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      EosParameters eosParams,
	      UserParameters userParams,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nX;
  (void) j; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceWest(const DataViewType& U,
	      int j,int k,int nQ,
	      Real x,Real y,Real z,Real t,
	      EosParameters eosParams,
	      UserParameters userParams,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) j; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

// Eastern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellEast(const DataViewType& U,int rad,int nX,
	      int j,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      EosParameters eosParams,
	      UserParameters userParams,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nX;
  (void) j; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceEast(const DataViewType& U,
	      int j,int k,int nQ,
	      Real x,Real y,Real z,Real t,
	      EosParameters eosParams,
	      UserParameters userParams,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) j; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

// Southern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellSouth(const DataViewType& U,int rad,int nY,
	       int i,int k,int nH,
	       Real x,Real y,Real z,Real t,
	       EosParameters eosParams,
	       UserParameters userParams,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;

  const Real xRef = cos(8.0*M_PI*z)/12.0 + 1.0/4.0;
  const Real pf = x>xRef ? -1.0 : 1.0; // Only reflect on part of the domain

  state[0] =    U(KFVM_D_DECL(i,rad + nH,k),0);
  state[1] =    U(KFVM_D_DECL(i,rad + nH,k),1);
  state[2] = pf*U(KFVM_D_DECL(i,rad + nH,k),2);
  state[3] =    U(KFVM_D_DECL(i,rad + nH,k),3);
  state[4] =    U(KFVM_D_DECL(i,rad + nH,k),4);
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceSouth(const DataViewType& U,
	       int i,int k,int nQ,
	       Real x,Real y,Real z,Real t,
	       EosParameters eosParams,
	       UserParameters userParams,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;

  const Real xRef = cos(8.0*M_PI*z)/12.0 + 1.0/4.0;
  const Real pf = x>xRef ? -1.0 : 1.0; // Only reflect on part of the boundary

  state[0] =    U(KFVM_DM_DECL(i,k),1,nQ,0);
  state[1] =    U(KFVM_DM_DECL(i,k),1,nQ,1);
  state[2] = pf*U(KFVM_DM_DECL(i,k),1,nQ,2);
  state[3] =    U(KFVM_DM_DECL(i,k),1,nQ,3);
  state[4] =    U(KFVM_DM_DECL(i,k),1,nQ,4);
}

// Northern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellNorth(const DataViewType& U,int rad,int nY,
	       int i,int k,int nH,
	       Real x,Real y,Real z,Real t,
	       EosParameters eosParams,
	       UserParameters userParams,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;

  const Real gam = eosParams.gamma,gamm = gam - 1.0;
  
  if(x < 8.66025*gam*t + 0.744017) {
    state[0] = 8.0;
    state[1] = 8.0*7.1447096;
    state[2] = -8.0*4.125;
    state[3] = 0.0;
    state[4] = 116.5/gamm + 4.0*(7.1447096*7.1447096 + 4.125*4.125);
  } else {
    state[0] = 1.4;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = 0.0;
    state[4] = 1.0/gamm;
  }
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceNorth(const DataViewType& U,
	       int i,int k,int nQ,
	       Real x,Real y,Real z,Real t,
	       EosParameters eosParams,
	       UserParameters userParams,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;

  const Real gam = eosParams.gamma,gamm = gam - 1.0;
  
  if(x < 8.66025*gam*t + 0.744017) {
    state[0] = 8.0;
    state[1] = 8.0*7.1447096;
    state[2] = -8.0*4.125;
    state[3] = 0.0;
    state[4] = 116.5/gamm + 4.0*(7.1447096*7.1447096 + 4.125*4.125);
  } else {
    state[0] = 1.4;
    state[1] = 0.0;
    state[2] = 0.0;
    state[3] = 0.0;
    state[4] = 1.0/gamm;
  }
}

// Bottom boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellBottom(const DataViewType& U,int rad,int nZ,
		int i,int j,int nH,
		Real x,Real y,Real z,Real t,
		EosParameters eosParams,
		UserParameters userParams,
		Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nZ;
  (void) i; (void) j; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceBottom(const DataViewType& U,
		int i,int j,int nQ,
		Real x,Real y,Real z,Real t,
		EosParameters eosParams,
		UserParameters userParams,
		Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) j; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

// Top boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellTop(const DataViewType& U,int rad,int nZ,
	     int i,int j,int nH,
	     Real x,Real y,Real z,Real t,
	     EosParameters eosParams,
	     UserParameters userParams,
	     Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nZ;
  (void) i; (void) j; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceTop(const DataViewType& U,
	     int i,int j,int nQ,
	     Real x,Real y,Real z,Real t,
	     EosParameters eosParams,
	     UserParameters userParams,
	     Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) j; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) eosParams; (void) userParams;
  (void) state;
}
