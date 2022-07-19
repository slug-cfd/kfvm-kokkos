#ifndef BOUNDARYCONDITIONS_K_H_
#define BOUNDARYCONDITIONS_K_H_

// File: BoundaryConditions_K.H
// Purpose: Define various boundary conditions for cell and face data

#include "Dimension.H"
#include <Kokkos_Macros.hpp>

#include <Definitions.H>
#include <SimVar.H>
#include <Types.H>
#include <Geometry.H>
#include <Numeric.H>
#include <BoundaryConditions.H>

namespace KFVM {

  namespace BoundaryConditions {

    // Unspecialized templates
    template<class DataViewType,BCType bcType>
    struct CellBcWest_K {};
    template<class DataViewType,BCType bcType>
    struct CellBcEast_K {};
    template<class DataViewType,BCType bcType>
    struct CellBcSouth_K {};
    template<class DataViewType,BCType bcType>
    struct CellBcNorth_K {};
    template<class DataViewType,BCType bcType>
    struct CellBcBottom_K {};
    template<class DataViewType,BCType bcType>
    struct CellBcTop_K {};
    
    template<class DataViewType,BCType bcType>
    struct FaceBcWest_K {};
    template<class DataViewType,BCType bcType>
    struct FaceBcEast_K {};
    template<class DataViewType,BCType bcType>
    struct FaceBcSouth_K {};
    template<class DataViewType,BCType bcType>
    struct FaceBcNorth_K {};
    template<class DataViewType,BCType bcType>
    struct FaceBcBottom_K {};
    template<class DataViewType,BCType bcType>
    struct FaceBcTop_K {};

    // Bring in user defined BCs
    namespace {
#include "UserBCs.H"
    }
    
    // ---------------- Western Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcWest_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nX;

      CellBcWest_K(const DataViewType& v_,int rad_,int nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,nV) = v(rad,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,k,nV) = v(rad,j,k,nV);
	  }
	}
      }
    };
      
    template<class DataViewType>
    struct CellBcWest_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nX;

      CellBcWest_K(const DataViewType& v_,int rad_,int nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(rad - nH - 1,j,k,nV) = pf*v(rad + nH,j,k,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcWest_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int rad,nX;

      CellBcWest_K(const DataViewType& v_,int rad_,int nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
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
    struct CellBcWest_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nX;
      Real t;

      CellBcWest_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nX_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nX(nX_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(-nH-1);
	  Real y = geom.yCoord(j - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellWest(v,rad,nX,j,0,nH,x,y,geom.zLo,t,state);
	  // Copy into view
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(-nH-1);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.yCoord(k - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellWest(v,rad,nX,j,k,nH,x,y,z,t,state);
	  // Copy into view
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(rad - nH - 1,j,k,nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Eastern Cell BC definitions. ---------------------------------    
    template<class DataViewType>
    struct CellBcEast_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nX;

      CellBcEast_K(const DataViewType& v_,int rad_,int nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,nV) = v(nX + rad - 1,j,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,k,nV) = v(nX + rad - 1,j,k,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcEast_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nX;

      CellBcEast_K(const DataViewType& v_,int rad_,int nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
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
    struct CellBcEast_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nX;
      Real t;

      CellBcEast_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nX_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nX(nX_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(nX + nH);
	  Real y = geom.yCoord(j - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellEast(v,rad,nX,j,0,nH,x,y,geom.zLo,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(nX + nH);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.yCoord(k - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellEast(v,rad,nX,j,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + rad + nH,j,k,nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Southern Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcSouth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nY;
      
      CellBcSouth_K(const DataViewType& v_,int rad_,int nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,nV) = v(i,rad,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,k,nV) = v(i,rad,k,nV);
	  }
	}
      }
    };

    template<class DataViewType>
    struct CellBcSouth_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nY;

      CellBcSouth_K(const DataViewType& v_,int rad_,int nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,rad - nH - 1,k,nV) = pf*v(i,rad + nH,k,nV);
	  }
	}
      }
    };

    template<class DataViewType>
    struct CellBcSouth_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int rad,nY;

      CellBcSouth_K(const DataViewType& v_,int rad_,int nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
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
    struct CellBcSouth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nY;
      Real t;

      CellBcSouth_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nY_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nY(nY_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(-nH - 1);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellSouth(v,rad,nY,i,0,nH,x,y,geom.zLo,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(-nH - 1);
	  Real z = geom.zCoord(k - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellSouth(v,rad,nY,i,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,rad - nH - 1,k,nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Northern Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcNorth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nY;

      CellBcNorth_K(const DataViewType& v_,int rad_,int nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,nV) = v(i,nY + rad - 1,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,k,nV) = v(i,nY + rad - 1,k,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcNorth_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nY;

      CellBcNorth_K(const DataViewType& v_,int rad_,int nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
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
    struct CellBcNorth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nY;
      Real t;

      CellBcNorth_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nY_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nY(nY_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(nY + nH);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellNorth(v,rad,nY,i,0,nH,x,y,geom.zLo,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(nY + nH);
	  Real z = geom.zCoord(k - rad);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellNorth(v,rad,nY,i,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + rad + nH,k,nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Bottom Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcBottom_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nZ;

      CellBcBottom_K(const DataViewType& v_,int rad_,int nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1,nV) = v(i,j,rad,nV);
	  } 
	}
      }
    };

    template<class DataViewType>
    struct CellBcBottom_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nZ;

      CellBcBottom_K(const DataViewType& v_,int rad_,int nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
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
    struct CellBcBottom_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int rad,nZ;

      CellBcBottom_K(const DataViewType& v_,int rad_,int nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1 ,nV) = v(i,j,nZ + rad - nH - 1,nV);
	    v(i,j,nZ + rad + nH,nV) = v(i,j,rad + nH         ,nV);
	  } 
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcBottom_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nZ;
      Real t;

      CellBcBottom_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nZ_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nZ(nZ_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.zCoord(-nH - 1);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellBottom(v,rad,nZ,i,j,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1 ,nV) = state[nV];
	  } 
	}
      }
    };
    
    // ---------------- Top Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcTop_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int rad,nZ;

      CellBcTop_K(const DataViewType& v_,int rad_,int nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = v(i,j,nZ + rad - 1,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcTop_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int rad,nZ;

      CellBcTop_K(const DataViewType& v_,int rad_,int nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
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
    struct CellBcTop_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      int rad,nZ;
      Real t;

      CellBcTop_K(const DataViewType& v_,const Geometry& geom_,int rad_,int nZ_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nZ(nZ_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over halo
	for (int nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.zCoord(nZ + nH);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellTop(v,rad,nZ,i,j,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = state[nV];
	  } 
	}
      }
    };
    
    // ---------------- West Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nX;

      FaceBcWest_K(const DataViewType& v_,int nX_):
	v(v_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0,j,FaceLabel::east,nQ,nV) = v(1,j,FaceLabel::west,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0,j,k,FaceLabel::east,nQ,nV) = v(1,j,k,FaceLabel::west,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nX;

      FaceBcWest_K(const DataViewType& v_,int nX_):
	v(v_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(0,j,k,FaceLabel::east,nQ,nV) = pf*v(1,j,k,FaceLabel::west,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int nX;

      FaceBcWest_K(const DataViewType& v_,int nX_):
	v(v_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0     ,j,k,FaceLabel::east,nQ,nV) = v(nX,j,k,FaceLabel::east,nQ,nV);
	    v(nX + 1,j,k,FaceLabel::west,nQ,nV) = v(1 ,j,k,FaceLabel::west,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nX;
      Real t;

      FaceBcWest_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nX_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nX(nX_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceWest(v,nX,j,0,nQ,geom.xLo,y,geom.zLo,t,state);
	  // Copy state over
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(0,j,FaceLabel::east,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceWest(v,nX,j,k,nQ*NUM_QUAD_PTS + nR,geom.xLo,y,z,t,state);
	    // Copy state over
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(0,j,k,FaceLabel::east,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- East Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcEast_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nX;

      FaceBcEast_K(const DataViewType& v_,int nX_):
	v(v_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + 1,j,FaceLabel::west,nQ,nV) = v(nX,j,FaceLabel::east,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + 1,j,k,FaceLabel::west,nQ,nV) = v(nX,j,k,FaceLabel::east,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcEast_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nX;

      FaceBcEast_K(const DataViewType& v_,int nX_):
	v(v_),
	nX(nX_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
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
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momx ? -1.0 : 1.0;
	    v(nX + 1,j,k,FaceLabel::west,nQ,nV) = pf*v(nX,j,k,FaceLabel::east,nQ,nV);
	  }
	}
      }
      
    };

    template<class DataViewType>
    struct FaceBcEast_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nX;
      Real t;

      FaceBcEast_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nX_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nX(nX_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceEast(v,nX,j,0,nQ,geom.xUp,y,geom.zLo,t,state);
	  // Copy state over
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(nX + 1,j,FaceLabel::west,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int j,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceEast(v,nX,j,k,nQ*NUM_QUAD_PTS + nR,geom.xUp,y,z,t,state);
	    // Copy state over
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(nX + 1,j,k,FaceLabel::west,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- South Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nY;

      FaceBcSouth_K(const DataViewType& v_,int nY_):
	v(v_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0,FaceLabel::north,nQ,nV) = v(i,1,FaceLabel::south,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0,k,FaceLabel::north,nQ,nV) = v(i,1,k,FaceLabel::south,nQ,nV);	    
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nY;

      FaceBcSouth_K(const DataViewType& v_,int nY_):
	v(v_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,0,k,FaceLabel::north,nQ,nV) = pf*v(i,1,k,FaceLabel::south,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int nY;

      FaceBcSouth_K(const DataViewType& v_,int nY_):
	v(v_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0     ,k,FaceLabel::north,nQ,nV) = v(i,nY,k,FaceLabel::north,nQ,nV);
	    v(i,nY + 1,k,FaceLabel::south,nQ,nV) = v(i,1 ,k,FaceLabel::south,nQ,nV);
	  }
	}
      }      
    };
    
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nY;
      Real t;

      FaceBcSouth_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nY_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nY(nY_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceSouth(v,nY,i,0,nQ,x,geom.yLo,geom.zLo,t,state);
	  // Copy into view
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,0,FaceLabel::north,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceSouth(v,nY,i,k,nQ*NUM_QUAD_PTS + nR,x,geom.yLo,z,t,state);
	    // Copy into view
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,0,k,FaceLabel::north,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- North Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcNorth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nY;

      FaceBcNorth_K(const DataViewType& v_,int nY_):
	v(v_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + 1,FaceLabel::south,nQ,nV) = v(i,nY,FaceLabel::north,nQ,nV);
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + 1,k,FaceLabel::south,nQ,nV) = v(i,nY,k,FaceLabel::north,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcNorth_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nY;

      FaceBcNorth_K(const DataViewType& v_,int nY_):
	v(v_),
	nY(nY_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
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
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momy ? -1.0 : 1.0;
	    v(i,nY + 1,k,FaceLabel::south,nQ,nV) = pf*v(i,nY,k,FaceLabel::north,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcNorth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nY;
      Real t;

      FaceBcNorth_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nY_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nY(nY_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceNorth(v,nY,i,0,nQ,x,geom.yUp,geom.zLo,t,state);
	  // copy state over
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,nY + 1,FaceLabel::south,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int k) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceNorth(v,nY,i,0,nQ,x,geom.yUp,z,t,state);
	    // copy state over
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,nY + 1,k,FaceLabel::south,nQ,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- Bottom Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nZ;

      FaceBcBottom_K(const DataViewType& v_,int nZ_):
	v(v_),
	nZ(nZ_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,0,FaceLabel::top,nQ,nV) = v(i,j,1,FaceLabel::bottom,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nZ;

      FaceBcBottom_K(const DataViewType& v_,int nZ_):
	v(v_),
	nZ(nZ_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	    v(i,j,0,FaceLabel::top,nQ,nV) = pf*v(i,j,1,FaceLabel::bottom,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::periodic> {
      DataViewType v;
      int nZ;

      FaceBcBottom_K(const DataViewType& v_,int nZ_):
	v(v_),
	nZ(nZ_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + 1,FaceLabel::bottom,nQ,nV) = v(i,j,1 ,FaceLabel::bottom,nQ,nV);
	    v(i,j,0     ,FaceLabel::top   ,nQ,nV) = v(i,j,nZ,FaceLabel::top   ,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nZ;
      Real t;

      FaceBcBottom_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nZ_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nZ(nZ_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceBottom(v,nZ,i,j,nQ*NUM_QUAD_PTS + nR,x,y,geom.zLo,t,state);
	    // copy state over
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,0,FaceLabel::top,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- Top Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcTop_K<DataViewType,BCType::outflow> {
      DataViewType v;
      int nZ;

      FaceBcTop_K(const DataViewType& v_,int nZ_):
	v(v_),
	nZ(nZ_) {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + 1,FaceLabel::bottom,nQ,nV) = v(i,j,nZ,FaceLabel::top,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcTop_K<DataViewType,BCType::reflecting> {
      DataViewType v;
      int nZ;

      FaceBcTop_K(const DataViewType& v_,int nZ_):
	v(v_),
	nZ(nZ_) {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Real pf = nV == SimVar::momz ? -1.0 : 1.0;
	    v(i,j,nZ + 1,FaceLabel::bottom,nQ,nV) = pf*v(i,j,nZ,FaceLabel::top,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcTop_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      int nZ;
      Real t;

      FaceBcTop_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,int nZ_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	nZ(nZ_),
	t(t_) {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over quadrature points
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceBottom(v,nZ,i,j,nQ*NUM_QUAD_PTS + nR,x,y,geom.zUp,t,state);
	    // copy state over
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,nZ + 1,FaceLabel::bottom,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };

  } // end namespace BoundaryConditions
    
} // end namespace KFVM

#endif
