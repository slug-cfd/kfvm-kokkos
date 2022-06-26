#ifndef BOUNDARYCONDITIONS_K_H_
#define BOUNDARYCONDITIONS_K_H_

// File: BoundaryConditions_K.H
// Purpose: Define various boundary conditions for cell and face data

#include <Kokkos_Macros.hpp>

#include "Types.H"

namespace KFVM {
  
  template<class DataViewType>
  struct CellBcEW_K {
    DataViewType v;
    idx_t rad,nX;

    CellBcEW_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
      v(v_),
      rad(rad_),
      nX(nX_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const {
      // Loop over SimVars
      for (int nV=0; nV<NUM_VARS; nV++) {
	v(rad - 1 ,j,nV) = v(rad         ,j,nV);
	v(nX + rad,j,nV) = v(nX + rad - 1,j,nV);
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int j,const int k) const {
      // Loop over SimVars
      for (int nV=0; nV<NUM_VARS; nV++) {
	v(rad - 1 ,j,k,nV) = v(rad         ,j,k,nV);
	v(nX + rad,j,k,nV) = v(nX + rad - 1,j,k,nV);
      }
    }
  };
  
  template<class DataViewType>
  struct CellBcNS_K {
    DataViewType v;
    idx_t rad,nY;

    CellBcNS_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
      v(v_),
      rad(rad_),
      nY(nY_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      // Loop over SimVars
      for (int nV=0; nV<NUM_VARS; nV++) {
	v(i,rad - 1 ,nV) = v(i,rad         ,nV);
	v(i,nY + rad,nV) = v(i,nY + rad - 1,nV);
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i,const int k) const {
      // Loop over SimVars
      for (int nV=0; nV<NUM_VARS; nV++) {
	v(i,rad - 1 ,k,nV) = v(i,rad         ,k,nV);
	v(i,nY + rad,k,nV) = v(i,nY + rad - 1,k,nV);
      }
    }
  };

  template<class DataViewType>
  struct CellBcTB_K {
    DataViewType v;
    idx_t rad,nZ;

    CellBcTB_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
      v(v_),
      rad(rad_),
      nZ(nZ_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i,const int j) const {
      // Loop over SimVars
      for (int nV=0; nV<NUM_VARS; nV++) {
	v(i,j,rad - 1 ,nV) = v(i,j,rad         ,nV);
	v(i,j,nZ + rad,nV) = v(i,j,nZ + rad - 1,nV);
      } 
    }
  };
  
  template<class DataViewType>
  struct FaceBcEW_K {
    DataViewType v;
    idx_t rad,nX;

    FaceBcEW_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
      v(v_),
      rad(rad_),
      nX(nX_) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int j) const {
      // Loop over quadrature points
      for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  v(0     ,j,FaceLabel::east,nQ,nV) = v(1 ,j,FaceLabel::west,nQ,nV);
	  v(nX + 1,j,FaceLabel::west,nQ,nV) = v(nX,j,FaceLabel::east,nQ,nV);
	}
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int j,const int k) const {
      // Loop over quadrature points
      for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0     ,j,k,FaceLabel::east,nQ,nR,nV) = v(1 ,j,k,FaceLabel::west,nQ,nR,nV);
	    v(nX + 1,j,k,FaceLabel::west,nQ,nR,nV) = v(nX,j,k,FaceLabel::east,nQ,nR,nV);
	  }
	}
      }
    }
  };
  
  template<class DataViewType>
  struct FaceBcNS_K {
    DataViewType v;
    idx_t rad,nY;

    FaceBcNS_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
      v(v_),
      rad(rad_),
      nY(nY_) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i) const {
      // Loop over quadrature points
      for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  v(i,0     ,FaceLabel::north,nQ,nV) = v(i,1 ,FaceLabel::south,nQ,nV);
	  v(i,nY + 1,FaceLabel::south,nQ,nV) = v(i,nY,FaceLabel::north,nQ,nV);
	}
      }
    }
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i,const int k) const {
      // Loop over quadrature points
      for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0     ,k,FaceLabel::north,nQ,nR,nV) = v(i,1 ,k,FaceLabel::south,nQ,nR,nV);
	    v(i,nY + 1,k,FaceLabel::south,nQ,nR,nV) = v(i,nY,k,FaceLabel::north,nQ,nR,nV);
	  }
	}
      }
    }
  };
  
  template<class DataViewType>
  struct FaceBcTB_K {
    DataViewType v;
    idx_t rad,nZ;

    FaceBcTB_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
      v(v_),
      rad(rad_),
      nZ(nZ_) {}
    
    KOKKOS_INLINE_FUNCTION
    void operator()(const int i,const int j) const {
      // Loop over quadrature points
      for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,0     ,FaceLabel::top   ,nQ,nR,nV) = v(i,j,1 ,FaceLabel::bottom,nQ,nR,nV);
	    v(i,j,nZ + 1,FaceLabel::bottom,nQ,nR,nV) = v(i,j,nZ,FaceLabel::top   ,nQ,nR,nV);
	  }
	}
      }
    }
  };
    
} // end namespace KFVM

#endif
