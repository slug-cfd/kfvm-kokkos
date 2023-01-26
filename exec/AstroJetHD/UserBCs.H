// This holds all user defined boundary conditions
// All of these functions must exist, regardless of BC type being used,
// however they can be left empty and will be ignored when using the
// built in ones

// There are two BC functions for each boundary, one to fill ghost cells
// prior to reconstrution, and one to fill face values priot to calling
// the Riemann solver

// Western boundary
#include "Definitions.H"

#define ASTROVEL 800

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellWest(const DataViewType& U,int rad,int nX,
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
void FaceWest(const DataViewType& U,
	      int j,int k,int nQ,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) j; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Eastern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellEast(const DataViewType& U,int rad,int nX,
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
void FaceEast(const DataViewType& U,
	      int j,int k,int nQ,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) j; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Southern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellSouth(const DataViewType& U,int rad,int nY,
	      int i,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real rsq = SPACE_DIM == 2 ? x*x : x*x + z*z;
  const Real Rsq = Real(0.05)*Real(0.05);
  const Real gam = Real(1.4),gamm = Real(0.4),vel = Real(ASTROVEL);

  if (rsq > Rsq) { // outflow
    state[0] = U(KFVM_D_DECL(i,rad + nH,k),0);
    state[1] = U(KFVM_D_DECL(i,rad + nH,k),1);
    state[2] = U(KFVM_D_DECL(i,rad + nH,k),2);
    state[3] = U(KFVM_D_DECL(i,rad + nH,k),3);
    state[4] = U(KFVM_D_DECL(i,rad + nH,k),4);
  } else { // Jet inflow
    state[0] = gam;
    state[1] = Real(0.0);
    state[2] = gam*vel;
    state[3] = Real(0.0);
    state[4] = Real(1.0)/gamm + gam*vel*vel/Real(2.0);
  }
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceSouth(const DataViewType& U,
	       int i,int k,int nQ,
	       Real x,Real y,Real z,Real t,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real rsq = SPACE_DIM == 2 ? x*x : x*x + z*z;
  const Real Rsq = Real(0.05)*Real(0.05);
  const Real gam = Real(1.4),gamm = Real(0.4),vel = Real(ASTROVEL);

  if (rsq > Rsq) { // outflow
    state[0] = U(KFVM_DM_DECL(i,k),1,nQ,0);
    state[1] = U(KFVM_DM_DECL(i,k),1,nQ,1);
    state[2] = U(KFVM_DM_DECL(i,k),1,nQ,2);
    state[3] = U(KFVM_DM_DECL(i,k),1,nQ,3);
    state[4] = U(KFVM_DM_DECL(i,k),1,nQ,4);
  } else { // Jet inflow
    state[0] = gam;
    state[1] = Real(0.0);
    state[2] = gam*vel;
    state[3] = Real(0.0);
    state[4] = Real(1.0)/gamm + gam*vel*vel/Real(2.0);
  }
}

// Northern boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellNorth(const DataViewType& U,int rad,int nY,
	      int i,int k,int nH,
	      Real x,Real y,Real z,Real t,
	      Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceNorth(const DataViewType& U,
	       int i,int k,int nQ,
	       Real x,Real y,Real z,Real t,
	       Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) k; (void) nQ;
  (void) y; (void) z;
  (void) state;
}

// Bottom boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellBottom(const DataViewType& U,int rad,int nZ,
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
void FaceBottom(const DataViewType& U,
		int i,int j,int nQ,
		Real x,Real y,Real z,Real t,
		Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) j; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}

// Top boundary
template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void CellTop(const DataViewType& U,int rad,int nZ,
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
void FaceTop(const DataViewType& U,
	     int i,int j,int nQ,
	     Real x,Real y,Real z,Real t,
	     Real state[NUM_VARS])
{
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) j; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;
}
