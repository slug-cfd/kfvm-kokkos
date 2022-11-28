#ifndef NUMERIC_K_H_
#define NUMERIC_K_H_

// File: Numeric_K.H
// Purpose: Define compute kernels for various simple numerical methods
//          referencing data from Numeric.H

#include <cstdint>
#include <cstdio>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"
#include "../ProblemSetup.H"
#include "../Geometry.H"
#include "../physics/Physics_K.H"
#include "Numeric.H"

namespace KFVM {

  namespace {
#include "InitialCondition.H"
  }

  namespace Numeric {

    template<class DataViewType>
    struct IntegrateIC_K
    {
      DataViewType v;
      QuadRuleView ab,wt;
      Geometry geom;

      IntegrateIC_K(const DataViewType& v_,
		    const QuadRuleView& ab_,
		    const QuadRuleView& wt_,
		    const Geometry& geom_):
	v(v_),
	ab(ab_),
	wt(wt_),
	geom(geom_) {}

      // 2D version
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {	
	// Center of cell (i,j)
	Real xCtr = geom.xCoord(i),yCtr = geom.yCoord(j);

	// Zero out state
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  v(i,j,nV) = 0.0;
	}

	// Loop over quadrature points and accumulate integral
	for (idx_t nX=0; nX<NUM_QUAD_PTS; nX++) {
	  for (idx_t nY=0; nY<NUM_QUAD_PTS; nY++) {
	    // Quadrature point
	    Real abX = sclAb(xCtr,geom.dx,ab(nX));
	    Real abY = sclAb(yCtr,geom.dy,ab(nY));

	    // IC at quadrature point
	    Real state[NUM_VARS];
	    InitialCondition(abX,abY,geom.zLo,state);

	    // Add each component into IC
	    for (idx_t nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,nV) += wt(nX)*wt(nY)*state[nV];
	    }
	  }
	}
      }

      // 3D version
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const {
	// Center of cell (i,j,k)
	Real xCtr = geom.xCoord(i),yCtr = geom.yCoord(j),zCtr = geom.zCoord(k);

	// Zero out state
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  v(i,j,k,nV) = 0.0;
	}

	// Loop over quadrature points and accumulate integral
	for (idx_t nX=0; nX<NUM_QUAD_PTS; nX++) {
	  for (idx_t nY=0; nY<NUM_QUAD_PTS; nY++) {
	    for (idx_t nZ=0; nZ<NUM_QUAD_PTS; nZ++) {
	      // Quadrature point
	      Real abX = sclAb(xCtr,geom.dx,ab(nX));
	      Real abY = sclAb(yCtr,geom.dy,ab(nY));
	      Real abZ = sclAb(zCtr,geom.dz,ab(nZ));

	      // IC at quadrature point
	      Real state[NUM_VARS];
	      InitialCondition(abX,abY,abZ,state);

	      // Add each component into IC
	      for (idx_t nV=0; nV<NUM_VARS; nV++) {
		v(i,j,k,nV) += wt(nX)*wt(nY)*wt(nZ)*state[nV];
	      }
	    }
	  }
	}
      }
    };

    template<class RhsViewType>
    struct IntegrateFlux_K
    {
      RhsViewType rhs;
      FaceDataView KFVM_D_DECL(flX,flY,flZ);
      QuadRuleView ab,wt;
      Geometry geom;

      IntegrateFlux_K(const RhsViewType& rhs_,
		      KFVM_D_DECL(const FaceDataView& flX_,
				  const FaceDataView& flY_,
				  const FaceDataView& flZ_),
		      const QuadRuleView& ab_,
		      const QuadRuleView& wt_,
		      const Geometry& geom_):
	rhs(rhs_),
	KFVM_D_DECL(flX(flX_),flY(flY_),flZ(flZ_)),
	ab(ab_),
	wt(wt_),
	geom(geom_) {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const {
	// Zero out RHS before accumulating fluxes
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  rhs(i,j,nV) = 0.0;
	}
			   
	// Loop over quadrature points and components
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    // x-direction
	    rhs(i,j,nV) += wt(nQ)*(flX(i,j,1,nQ,nV) - flX(i + 1,j,0,nQ,nV))/geom.dx;
	    // y-direction
	    rhs(i,j,nV) += wt(nQ)*(flY(i,j,1,nQ,nV) - flY(i,j + 1,0,nQ,nV))/geom.dy;
	  }
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const {
	// Zero out RHS before accumulating fluxes
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  rhs(i,j,k,nV) = 0.0;
	}
			   
	// Loop over quadrature points and components
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (idx_t nR=0; nR<NUM_QUAD_PTS; nR++) {
	      idx_t qIdx = nQ*NUM_QUAD_PTS + nR;
	      Real qWt = wt(nQ)*wt(nR);
	      rhs(i,j,k,nV) += qWt*(flX(i,j,k,0,qIdx,nV) -
				    flX(i + 1,j,k,1,qIdx,nV))/geom.dx;
	      rhs(i,j,k,nV) += qWt*(flY(i,j,k,0,qIdx,nV) -
				    flY(i,j + 1,k,1,qIdx,nV))/geom.dy;
	      rhs(i,j,k,nV) += qWt*(flZ(i,j,k,0,qIdx,nV) -
				    flZ(i,j,k + 1,1,qIdx,nV))/geom.dz;
	    }
	  }
	}
      }
#endif
    };

    template<class UViewType,class KViewType,int nStage>
    struct SSP45_S1_K {
      UViewType U1,U;
      KViewType K;
      SSPCoeff<nStage> rk;
      Real dt;

      SSP45_S1_K(const UViewType& U1_,
		 const UViewType& U_,
		 const KViewType& K_,
		 const SSPCoeff<nStage>& rk_,
		 Real dt_):
	U1(U1_),
	U(U_),
	K(K_),
	rk(rk_),
	dt(dt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  U1(KFVM_D_DECL(i,j,k),nV) =
	    rk.a_1_0*U(KFVM_D_DECL(i,j,k),nV) + dt*rk.b_1*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType,int nStage>
    struct SSP45_S2_K {
      UViewType U2,U,U1;
      KViewType K;
      SSPCoeff<nStage> rk;
      Real dt;

      SSP45_S2_K(const UViewType& U2_,
		 const UViewType& U_,
		 const UViewType& U1_,
		 const KViewType& K_,
		 const SSPCoeff<nStage>& rk_,
		 Real dt_):
	U2(U2_),
	U(U_),
	U1(U1_),
	K(K_),
	rk(rk_),
	dt(dt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  U2(KFVM_D_DECL(i,j,k),nV) =
	    rk.a_2_0*U(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_2_1*U1(KFVM_D_DECL(i,j,k),nV) +
	    dt*rk.b_2*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType,int nStage>
    struct SSP45_S3_K {
      UViewType U3,U,U2;
      KViewType K;
      SSPCoeff<nStage> rk;
      Real dt;

      SSP45_S3_K(const UViewType& U3_,
		 const UViewType& U_,
		 const UViewType& U2_,
		 const KViewType& K_,
		 const SSPCoeff<nStage>& rk_,
		 Real dt_):
	U3(U3_),
	U(U_),
	U2(U2_),
	K(K_),
	rk(rk_),
	dt(dt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  U3(KFVM_D_DECL(i,j,k),nV) =
	    rk.a_3_0*U(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_3_2*U2(KFVM_D_DECL(i,j,k),nV) +
	    dt*rk.b_3*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType,int nStage>
    struct SSP45_S4_K {
      UViewType U4,U,U3;
      KViewType K;
      SSPCoeff<nStage> rk;
      Real dt;

      SSP45_S4_K(const UViewType& U4_,
		 const UViewType& U_,
		 const UViewType& U3_,
		 const KViewType& K_,
		 const SSPCoeff<nStage>& rk_,
		 Real dt_):
	U4(U4_),
	U(U_),
	U3(U3_),
	K(K_),
	rk(rk_),
	dt(dt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  U4(KFVM_D_DECL(i,j,k),nV) =
	    rk.a_4_0*U(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_4_3*U3(KFVM_D_DECL(i,j,k),nV) +
	    dt*rk.b_4*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType,class WViewType,int nStage>
    struct SSP45_S5_K {
      UViewType U,U2,U3,U4;
      KViewType K,Ktil;
      WViewType W;
      SSPCoeff<nStage> rk;
      Real dt;

      SSP45_S5_K(const UViewType& U_,
		 const UViewType& U2_,
		 const UViewType& U3_,
		 const UViewType& U4_,
		 const KViewType& K_,
		 const KViewType& Ktil_,
		 const WViewType& W_,
		 const SSPCoeff<nStage>& rk_,
		 Real dt_):
	U(U_),
	U2(U2_),
	U3(U3_),
	U4(U4_),
	K(K_),
	Ktil(Ktil_),
	W(W_),
	rk(rk_),
	dt(dt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	using Kokkos::Experimental::fabs;
	using Kokkos::Experimental::fmin;

	// Pull out a few states
	Real u0[NUM_VARS],u1[NUM_VARS],u2[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  u0[nV] = U(KFVM_D_DECL(i,j,k),nV);
	  u1[nV] = U3(KFVM_D_DECL(i,j,k),nV);
	}

	// Update state
	for (idx_t nV=0; nV<NUM_VARS; nV++) {
	  U(KFVM_D_DECL(i,j,k),nV) =
	    rk.a_5_0*U(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_5_2*U2(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_5_3*U3(KFVM_D_DECL(i,j,k),nV) +
	    rk.a_5_4*U4(KFVM_D_DECL(i,j,k),nV) +
	    dt*rk.b_5*K(KFVM_D_DECL(i,j,k),nV) +
	    dt*rk.b_5til*Ktil(KFVM_D_DECL(i,j,k),nV);
	}

	// Finish smoothness indicator calculation
	for (int nV=0; nV<NUM_VARS; nV++) {
	  u2[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real e0 = Physics::EOS<eqType>::eint(u0);
	Real e2 = Physics::EOS<eqType>::eint(u2);
	W(KFVM_D_DECL(i,j,k)) = fabs(e2 - e0)/fmin(e0,e2);
      }
    };
    
  } // End numeric namespace
  
} // End KFVM namespace

#endif
