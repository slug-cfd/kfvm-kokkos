#ifndef BOUNDARYCONDITIONS_K_H_
#define BOUNDARYCONDITIONS_K_H_

// File: BoundaryConditions_K.H
// Purpose: Define various boundary conditions for cell and face data
#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Types.H"
#include "Geometry.H"
#include "numeric/Numeric.H"
#include "physics/Physics_K.H"

namespace KFVM {

  namespace BoundaryConditions {

    // Unspecialized templates
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcWest_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcEast_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcSouth_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcNorth_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcBottom_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct CellBcTop_K {};
    
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcWest_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcEast_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcSouth_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcNorth_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcBottom_K {};
    template<class DataViewType,BCType bcType,class RefCoeff = void>
    struct FaceBcTop_K {};

    // Bring in user defined BCs
    namespace {
#include "UserBCs.H"
    }
    
    // ---------------- Western Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcWest_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nX;

      CellBcWest_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(rad - nH - 1,j,k),nV) = v(KFVM_D_DECL(rad + nH,j,k),nV);
	  }
	}
      }
    };
      
    template<class DataViewType,class RefCoeff>
    struct CellBcWest_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nX;

      CellBcWest_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nX_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(rad - nH - 1,j,k),nV) = pf(nV)*v(KFVM_D_DECL(rad + nH,j,k),nV);
	  }
	}
      }
    };
      
    template<class DataViewType>
    struct CellBcWest_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nX;
      Real t;

      CellBcWest_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nX_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nX(nX_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(-nH-1);
	  Real y = geom.yCoord(j - rad);
#if (SPACE_DIM == 2)
	  const idx_t k = 0;
	  Real z = geom.zLo;
#else
	  Real z = geom.zCoord(k - rad);
#endif
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellWest(v,rad,nX,j,k,nH,x,y,z,t,state);
	  // Copy idx_to view
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(rad - nH - 1,j,k),nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Eastern Cell BC definitions. ---------------------------------    
    template<class DataViewType>
    struct CellBcEast_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nX;

      CellBcEast_K(const DataViewType& v_,idx_t rad_,idx_t nX_):
	v(v_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(nX + rad + nH,j,k),nV) = v(KFVM_D_DECL(nX + rad - nH - 1,j,k),nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct CellBcEast_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nX;

      CellBcEast_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nX_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nX(nX_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(nX + rad + nH,j,k),nV) = pf(nV)*v(KFVM_D_DECL(nX + rad - nH - 1,j,k),nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcEast_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nX;
      Real t;

      CellBcEast_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nX_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nX(nX_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(nX + nH);
	  Real y = geom.yCoord(j - rad);
#if (SPACE_DIM == 2)
	  const idx_t k = 0;
	  Real z = geom.zLo;
#else
	  Real z = geom.zCoord(k - rad);
#endif
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellEast(v,rad,nX,j,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(nX + rad + nH,j,k),nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Southern Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcSouth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nY;
      
      CellBcSouth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,rad - nH - 1,k),nV) = v(KFVM_D_DECL(i,rad + nH,k),nV);
	  }
	}
      }
    };

    template<class DataViewType,class RefCoeff>
    struct CellBcSouth_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nY;

      CellBcSouth_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nY_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,rad - nH - 1,k),nV) = pf(nV)*v(KFVM_D_DECL(i,rad + nH,k),nV);
	  }
	}
      }
    };

    template<class DataViewType>
    struct CellBcSouth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nY;
      Real t;

      CellBcSouth_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nY_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nY(nY_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(-nH - 1);
#if (SPACE_DIM == 2)
	  const idx_t k = 0;
	  Real z = geom.zLo;
#else
	  Real z = geom.zCoord(k - rad);
#endif
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellSouth(v,rad,nY,i,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,rad - nH - 1,k),nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Northern Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcNorth_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nY;

      CellBcNorth_K(const DataViewType& v_,idx_t rad_,idx_t nY_):
	v(v_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,nY + rad + nH,k),nV) = v(KFVM_D_DECL(i,nY + rad - nH - 1,k),nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct CellBcNorth_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nY;

      CellBcNorth_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nY_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nY(nY_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,nY + rad + nH,k),nV) = pf(nV)*v(KFVM_D_DECL(i,nY + rad - nH - 1,k),nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcNorth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nY;
      Real t;

      CellBcNorth_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nY_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nY(nY_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(nY + nH);
#if (SPACE_DIM == 2)
	  const idx_t k = 0;
	  Real z = geom.zLo;
#else
	  Real z = geom.zCoord(k - rad);
#endif
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellNorth(v,rad,nY,i,k,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_D_DECL(i,nY + rad + nH,k),nV) = state[nV];
	  }
	}
      }
    };
    
    // ---------------- Bottom Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcBottom_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nZ;

      CellBcBottom_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1,nV) = v(i,j,rad + nH,nV);
	  } 
	}
      }
    };

    template<class DataViewType,class RefCoeff>
    struct CellBcBottom_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nZ;

      CellBcBottom_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nZ_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1,nV) = pf(nV)*v(i,j,rad + nH,nV);
	  } 
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcBottom_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nZ;
      Real t;

      CellBcBottom_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nZ_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nZ(nZ_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.zCoord(-nH - 1);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellBottom(v,rad,nZ,i,j,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,rad - nH - 1 ,nV) = state[nV];
	  } 
	}
      }
    };
    
    // ---------------- Top Cell BC definitions. ---------------------------------
    template<class DataViewType>
    struct CellBcTop_K<DataViewType,BCType::outflow> {
      DataViewType v;
      idx_t rad,nZ;

      CellBcTop_K(const DataViewType& v_,idx_t rad_,idx_t nZ_):
	v(v_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = v(i,j,nZ + rad - nH - 1,nV);
	  } 
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct CellBcTop_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;
      idx_t rad,nZ;

      CellBcTop_K(const DataViewType& v_,const RefCoeff& pf_,idx_t rad_,idx_t nZ_):
	v(v_),
        pf(pf_),
	rad(rad_),
	nZ(nZ_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = pf(nV)*v(i,j,nZ + rad - nH - 1,nV);
	  } 
	}
      }
    };
    
    template<class DataViewType>
    struct CellBcTop_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      idx_t rad,nZ;
      Real t;

      CellBcTop_K(const DataViewType& v_,const Geometry& geom_,idx_t rad_,idx_t nZ_,Real t_):
	v(v_),
	geom(geom_),
	rad(rad_),
	nZ(nZ_),
	t(t_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over halo
	for (idx_t nH=0; nH<rad; nH++) {
	  // Get cell center in halo
	  Real x = geom.xCoord(i - rad);
	  Real y = geom.yCoord(j - rad);
	  Real z = geom.zCoord(nZ + nH);
	  // query user defined BCs
	  Real state[NUM_VARS];
	  CellTop(v,rad,nZ,i,j,nH,x,y,z,t,state);
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,nZ + rad + nH,nV) = state[nV];
	  } 
	}
      }
    };
    
    // ---------------- West Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcWest_K(const DataViewType& v_): v(v_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(j,k),0,nQ,nV) = v(KFVM_DM_DECL(j,k),1,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcWest_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcWest_K(const DataViewType& v_,const RefCoeff& pf_):
        v(v_),
        pf(pf_)
      {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(j,k),0,nQ,nV) = pf(nV)*v(KFVM_DM_DECL(j,k),1,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcWest_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcWest_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceWest(v,j,0,nQ,geom.xLo,y,geom.zLo,t,state);
	  // Copy state over
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(j,0,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t j,const idx_t k) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceWest(v,j,k,nQ*NUM_QUAD_PTS + nR,geom.xLo,y,z,t,state);
	    // Copy state over
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(j,k,0,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- East Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcEast_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcEast_K(const DataViewType& v_):
	v(v_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(j,k),1,nQ,nV) = v(KFVM_DM_DECL(j,k),0,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcEast_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcEast_K(const DataViewType& v_,const RefCoeff& pf_):
	v(v_),
        pf(pf_)
      {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t j,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(j,k),1,nQ,nV) = pf(nV)*v(KFVM_DM_DECL(j,k),0,nQ,nV);
	  }
	}
      }
      
    };

    template<class DataViewType>
    struct FaceBcEast_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcEast_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceEast(v,j,0,nQ,geom.xUp,y,geom.zLo,t,state);
	  // Copy state over
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(j,1,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t j,const idx_t k) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceEast(v,j,k,nQ*NUM_QUAD_PTS + nR,geom.xUp,y,z,t,state);
	    // Copy state over
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(j,k,1,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- South Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcSouth_K(const DataViewType& v_):
	v(v_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(i,k),0,nQ,nV) = v(KFVM_DM_DECL(i,k),1,nQ,nV);	    
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcSouth_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcSouth_K(const DataViewType& v_,const RefCoeff& pf_):
	v(v_),
        pf(pf_)
      {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(i,k),0,nQ,nV) = pf(nV)*v(KFVM_DM_DECL(i,k),1,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcSouth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcSouth_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceSouth(v,i,0,nQ,x,geom.yLo,geom.zLo,t,state);
	  // Copy idx_to view
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,0,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t k) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceSouth(v,i,k,nQ*NUM_QUAD_PTS + nR,x,geom.yLo,z,t,state);
	    // Copy idx_to view
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(i,k,0,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- North Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcNorth_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcNorth_K(const DataViewType& v_):
	v(v_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(i,k),1,nQ,nV) = v(KFVM_DM_DECL(i,k),0,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcNorth_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcNorth_K(const DataViewType& v_,const RefCoeff& pf_):
	v(v_),
        pf(pf_)
      {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_DM_DECL(const idx_t i,const idx_t k)) const {
	// Loop over quadrature points
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(KFVM_DM_DECL(i,k),1,nQ,nV) = pf(nV)*v(KFVM_DM_DECL(i,k),0,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcNorth_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcNorth_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
      
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // Get quadrature point global location
	  Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	  // query user defined BCs
	  Real state[NUM_VARS];
	  FaceNorth(v,i,0,nQ,x,geom.yUp,geom.zLo,t,state);
	  // copy state over
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,1,nQ,nV) = state[nV];
	  }
	}
      }
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t k) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real z = Numeric::sclAb(geom.zCoord(k),geom.dz,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceNorth(v,i,k,nQ*NUM_QUAD_PTS + nR,x,geom.yUp,z,t,state);
	    // copy state over
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(i,k,1,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- Bottom Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcBottom_K(const DataViewType& v_):
	v(v_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,0,nQ,nV) = v(i,j,1,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcBottom_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcBottom_K(const DataViewType& v_,const RefCoeff& pf_):
	v(v_),
        pf(pf_)
      {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,0,nQ,nV) = pf(nV)*v(i,j,1,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcBottom_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcBottom_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
    
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceBottom(v,i,j,nQ*NUM_QUAD_PTS + nR,x,y,geom.zLo,t,state);
	    // copy state over
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,0,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };
    
    // ---------------- Top Face BC definitions. ---------------------------------
    template<class DataViewType>
    struct FaceBcTop_K<DataViewType,BCType::outflow> {
      DataViewType v;

      FaceBcTop_K(const DataViewType& v_): v(v_) {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,1,nQ,nV) = v(i,j,0,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType,class RefCoeff>
    struct FaceBcTop_K<DataViewType,BCType::reflecting,RefCoeff> {
      DataViewType v;
      RefCoeff pf;

      FaceBcTop_K(const DataViewType& v_,const RefCoeff& pf_):
        v(v_),
        pf(pf_)
      {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Loop over SimVars
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    v(i,j,1,nQ,nV) = pf(nV)*v(i,j,0,nQ,nV);
	  }
	}
      }
    };
    
    template<class DataViewType>
    struct FaceBcTop_K<DataViewType,BCType::user> {
      DataViewType v;
      Geometry geom;
      QuadRuleView ab;
      Real t;

      FaceBcTop_K(const DataViewType& v_,const Geometry& geom_,const QuadRuleView& ab_,Real t_):
	v(v_),
	geom(geom_),
	ab(ab_),
	t(t_) {}
        
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Loop over quadrature points
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	    // Get quadrature point global location
	    Real x = Numeric::sclAb(geom.xCoord(i),geom.dx,ab(nQ));
	    Real y = Numeric::sclAb(geom.yCoord(j),geom.dy,ab(nR));
	    // query user defined BCs
	    Real state[NUM_VARS];
	    FaceTop(v,i,j,nQ*NUM_QUAD_PTS + nR,x,y,geom.zUp,t,state);
	    // copy state over
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,1,nQ*NUM_QUAD_PTS + nR,nV) = state[nV];
	    }
	  }
	}
      }
    };

  } // end namespace BoundaryConditions
    
} // end namespace KFVM

#endif
