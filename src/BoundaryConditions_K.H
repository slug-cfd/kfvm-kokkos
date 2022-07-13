#ifndef BOUNDARYCONDITIONS_K_H_
#define BOUNDARYCONDITIONS_K_H_

// File: BoundaryConditions_K.H
// Purpose: Define various boundary conditions for cell and face data

#include <Kokkos_Macros.hpp>

#include "SimVar.H"
#include "Types.H"

namespace KFVM {

  namespace BoundaryConditions {

    // Work tags for different BC types
    enum BCType {periodic = 0,outflow,reflecting,user};
    struct Periodic {};
    struct Outflow {};
    struct Reflecting {};
    struct User {}; // placeholder, not sure how to implement this one...
  
    template<class DataViewType>
    struct CellBcWest_K {
      DataViewType v;
      idx_t rad,nX;

      CellBcWest_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,nV) = v(rad,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,k,nV) = v(rad,j,k,nV);
	  }
	}
      }
      
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(rad - nH - 1,j,nV) = pf*v(rad + nH,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(rad - nH - 1,j,k,nV) = pf*v(rad + nH,j,k,nV);
	  }
	}
      }
      
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1 ,j,nV) = v(nX + rad - nH - 1,j,nV);
	    v(nX + rad + nH,j,nV) = v(rad + nH         ,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int j,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1 ,j,k,nV) = v(nX + rad - nH - 1,j,k,nV);
	    v(nX + rad + nH,j,k,nV) = v(rad + nH         ,j,k,nV);
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct CellBcEast_K {
      DataViewType v;
      idx_t rad,nX;

      CellBcEast_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,nV) = v(nX + rad - 1,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,k,nV) = v(nX + rad - 1,j,k,nV);
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(nX + rad + nH,j,nV) = pf*v(nX + rad - nH - 1,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(nX + rad + nH,j,k,nV) = pf*v(nX + rad - nH - 1,j,k,nV);
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct CellBcSouth_K {
      DataViewType v;
      idx_t rad,nY;

      CellBcSouth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,nV) = v(i,rad,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,k,nV) = v(i,rad,k,nV);
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,rad - nH - 1,nV) = pf*v(i,rad + nH,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,rad - nH - 1,k,nV) = pf*v(i,rad + nH,k,nV);
	  }
	}
      }
      
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1 ,nV) = v(i,nY + rad - nH - 1,nV);
	    v(i,nY + rad + nH,nV) = v(i,rad + nH         ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1 ,k,nV) = v(i,nY + rad - nH - 1,k,nV);
	    v(i,nY + rad + nH,k,nV) = v(i,rad + nH         ,k,nV);
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct CellBcNorth_K {
      DataViewType v;
      idx_t rad,nY;

      CellBcNorth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,nV) = v(i,nY + rad - 1,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,k,nV) = v(i,nY + rad - 1,k,nV);
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,nY + rad + nH,nV) = pf*v(i,nY + rad - nH - 1,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int k) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,nY + rad + nH,k,nV) = pf*v(i,nY + rad - nH - 1,k,nV);
	  }
	}
      }
    };

    template<class DataViewType>
    struct CellBcBottom_K {
      DataViewType v;
      idx_t rad,nZ;

      CellBcBottom_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1 ,nV) = v(i,j,nZ + rad - nH - 1,nV);
	    v(i,j,nZ + rad + nH,nV) = v(i,j,rad + nH         ,nV);
	  } 
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1,nV) = v(i,j,rad,nV);
	  } 
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	    v(i,j,rad - nH - 1,nV) = pf*v(i,j,rad + nH,nV);
	  } 
	}
      }
    };

    template<class DataViewType>
    struct CellBcTop_K {
      DataViewType v;
      idx_t rad,nZ;

      CellBcTop_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = v(i,j,nZ + rad - 1,nV);
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int j) const {
	(void) bct;
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	    v(i,j,nZ + rad + nH,nV) = pf*v(i,j,nZ + rad - nH - 1,nV);
	  } 
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcWest_K {
      DataViewType v;
      idx_t rad,nX;

      FaceBcWest_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0     ,j,FaceLabel::east,nQ,nV) = v(nX,j,FaceLabel::east,nQ,nV);
	    v(nX + 1,j,FaceLabel::west,nQ,nV) = v(1 ,j,FaceLabel::west,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int j,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(0     ,j,k,FaceLabel::east,nQ,nR,nV) = v(nX,j,k,FaceLabel::east,nQ,nR,nV);
	      v(nX + 1,j,k,FaceLabel::west,nQ,nR,nV) = v(1 ,j,k,FaceLabel::west,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0,j,FaceLabel::east,nQ,nV) = v(1,j,FaceLabel::west,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(0,j,k,FaceLabel::east,nQ,nR,nV) = v(1,j,k,FaceLabel::west,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(0,j,FaceLabel::east,nQ,nV) = pf*v(1,j,FaceLabel::west,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	      v(0,j,k,FaceLabel::east,nQ,nR,nV) = pf*v(1,j,k,FaceLabel::west,nQ,nR,nV);
	    }
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcEast_K {
      DataViewType v;
      idx_t rad,nX;

      FaceBcEast_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + 1,j,FaceLabel::west,nQ,nV) = v(nX,j,FaceLabel::east,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int j,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(nX + 1,j,k,FaceLabel::west,nQ,nR,nV) = v(nX,j,k,FaceLabel::east,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(nX + 1,j,FaceLabel::west,nQ,nV) = pf*v(nX,j,FaceLabel::east,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int j,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	      v(nX + 1,j,k,FaceLabel::west,nQ,nR,nV) = pf*v(nX,j,k,FaceLabel::east,nQ,nR,nV);
	    }
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcSouth_K {
      DataViewType v;
      idx_t rad,nY;

      FaceBcSouth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0     ,FaceLabel::north,nQ,nV) = v(i,nY,FaceLabel::north,nQ,nV);
	    v(i,nY + 1,FaceLabel::south,nQ,nV) = v(i,1 ,FaceLabel::south,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,0     ,k,FaceLabel::north,nQ,nR,nV) = v(i,nY,k,FaceLabel::north,nQ,nR,nV);
	      v(i,nY + 1,k,FaceLabel::south,nQ,nR,nV) = v(i,1 ,k,FaceLabel::south,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0,FaceLabel::north,nQ,nV) = v(i,1,FaceLabel::south,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,0,k,FaceLabel::north,nQ,nR,nV) = v(i,1,k,FaceLabel::south,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,0,FaceLabel::north,nQ,nV) = pf*v(i,1,FaceLabel::south,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	      v(i,0,k,FaceLabel::north,nQ,nR,nV) = pf*v(i,1,k,FaceLabel::south,nQ,nR,nV);
	    }
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcNorth_K {
      DataViewType v;
      idx_t rad,nY;

      FaceBcNorth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + 1,FaceLabel::south,nQ,nV) = v(i,nY,FaceLabel::north,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,nY + 1,k,FaceLabel::south,nQ,nR,nV) = v(i,nY,k,FaceLabel::north,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,nY + 1,FaceLabel::south,nQ,nV) = pf*v(i,nY,FaceLabel::north,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int k) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	      v(i,nY + 1,k,FaceLabel::south,nQ,nR,nV) = pf*v(i,nY,k,FaceLabel::north,nQ,nR,nV);
	    }
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcBottom_K {
      DataViewType v;
      idx_t rad,nZ;

      FaceBcBottom_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Periodic bct,const int i,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,nZ + 1,FaceLabel::bottom,nQ,nR,nV) = v(i,j,1 ,FaceLabel::bottom,nQ,nR,nV);
	      v(i,j,0     ,FaceLabel::top   ,nQ,nR,nV) = v(i,j,nZ,FaceLabel::top   ,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,0,FaceLabel::top,nQ,nR,nV) = v(i,j,1,FaceLabel::bottom,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	      v(i,j,0,FaceLabel::top,nQ,nR,nV) = pf*v(i,j,1,FaceLabel::bottom,nQ,nR,nV);
	    }
	  }
	}
      }
    };
  
    template<class DataViewType>
    struct FaceBcTop_K {
      DataViewType v;
      idx_t rad,nZ;

      FaceBcTop_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Outflow bct,const int i,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,nZ + 1,FaceLabel::bottom,nQ,nR,nV) = v(i,j,nZ,FaceLabel::top,nQ,nR,nV);
	    }
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(Reflecting bct,const int i,const int j) const {
	(void) bct;
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Loop over SimVars
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	      v(i,j,nZ + 1,FaceLabel::bottom,nQ,nR,nV) = pf*v(i,j,nZ,FaceLabel::top,nQ,nR,nV);
	    }
	  }
	}
      }
    };

  } // end namespace BoundaryConditions
    
} // end namespace KFVM

#endif
