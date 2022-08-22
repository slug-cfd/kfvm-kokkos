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
  (void) x; (void) y; (void) z; (void) t;
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

#endif
