// This holds all user defined boundary conditions
// All of these functions must exist, regardless of BC type being used,
// however they can be left empty and will be ignored when using the
// built in ones

// There are two BC functions for each boundary, one to fill ghost cells
// prior to reconstrution, and one to fill face values priot to calling
// the Riemann solver

#include "Definitions.H"
#ifndef B0SQ
#define B0SQ 2000.0
#endif

// Western boundary
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
  using Kokkos::Experimental::sqrt;
  // Void casts to ignore unused parameter warnings
  (void) U; (void) rad; (void) nY;
  (void) i; (void) k; (void) nH;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real rsq = SPACE_DIM == 2 ? x*x : x*x + z*z;
  const Real Rsq = 0.05*0.05,b0 = sqrt(B0SQ);

  if (rsq > Rsq) { // outflow
    state[0] = U(KFVM_D_DECL(i,rad + nH,k),0);
    state[1] = U(KFVM_D_DECL(i,rad + nH,k),1);
    state[2] = U(KFVM_D_DECL(i,rad + nH,k),2);
    state[3] = U(KFVM_D_DECL(i,rad + nH,k),3);
    state[4] = U(KFVM_D_DECL(i,rad + nH,k),4);
    state[5] = U(KFVM_D_DECL(i,rad + nH,k),5);
    state[6] = U(KFVM_D_DECL(i,rad + nH,k),6);
    state[7] = U(KFVM_D_DECL(i,rad + nH,k),7);
    state[8] = U(KFVM_D_DECL(i,rad + nH,k),8);
  } else { // Jet inflow
    state[0] = 1.4;
    state[1] = 0.0;
    state[2] = 800.0*1.4;
    state[3] = 0.0;
    state[4] = 1.0/0.4 + 1.4*800.0*800.0/2.0 + b0*b0/2.0;
    state[5] = 0.0;
    state[6] = sqrt(b0);
    state[7] = 0.0;
    state[8] = 0.0;
  }
}

template<class DataViewType>
KOKKOS_INLINE_FUNCTION
void FaceSouth(const DataViewType& U,
	       int i,int k,int nQ,
	       Real x,Real y,Real z,Real t,
	       Real state[NUM_VARS])
{
  using Kokkos::Experimental::sqrt;
  // Void casts to ignore unused parameter warnings
  (void) U;
  (void) i; (void) k; (void) nQ;
  (void) x; (void) y; (void) z; (void) t;
  (void) state;

  const Real rsq = SPACE_DIM == 2 ? x*x : x*x + z*z;
  const Real Rsq = 0.05*0.05,b0 = sqrt(B0SQ);

  if (rsq > Rsq) { // outflow
    state[0] = U(KFVM_DM_DECL(i,k),1,nQ,0);
    state[1] = U(KFVM_DM_DECL(i,k),1,nQ,1);
    state[2] = U(KFVM_DM_DECL(i,k),1,nQ,2);
    state[3] = U(KFVM_DM_DECL(i,k),1,nQ,3);
    state[4] = U(KFVM_DM_DECL(i,k),1,nQ,4);
    state[5] = U(KFVM_DM_DECL(i,k),1,nQ,5);
    state[6] = U(KFVM_DM_DECL(i,k),1,nQ,6);
    state[7] = U(KFVM_DM_DECL(i,k),1,nQ,7);
    state[8] = U(KFVM_DM_DECL(i,k),1,nQ,8);
  } else { // Jet inflow
    state[0] = 1.4;
    state[1] = 0.0;
    state[2] = 800.0*1.4;
    state[3] = 0.0;
    state[4] = 1.0/0.4 + 1.4*800.0*800.0/2.0 + b0*b0/2.0;
    state[5] = 0.0;
    state[6] = b0;
    state[7] = 0.0;
    state[8] = 0.0;
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
