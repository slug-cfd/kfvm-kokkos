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
	  v(i,j,nV) = Real(0.0);
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
	  v(i,j,k,nV) = Real(0.0);
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
	  rhs(i,j,nV) = Real(0.0);
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
	  rhs(i,j,k,nV) = Real(0.0);
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

    template<class UViewType,class KViewType>
    struct RKFSAL_StagePre_K {
      UViewType U,Uprev;
      KViewType Uhat,Utmp,K;
      Real betaDt,bhatDt;

      RKFSAL_StagePre_K(const UViewType& U_,
                        const KViewType& Uhat_,
                        const KViewType& Utmp_,
                        const UViewType& Uprev_,
                        const KViewType& K_,
                        Real betaDt_,
                        Real bhatDt_):
	U(U_),
	Uprev(Uprev_),
	Uhat(Uhat_),
	Utmp(Utmp_),
	K(K_),
	betaDt(betaDt_),
	bhatDt(bhatDt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (int nV=0; nV<NUM_VARS; nV++) {
	  U(KFVM_D_DECL(i,j,k),nV) = Uprev(KFVM_D_DECL(i,j,k),nV) + betaDt*K(KFVM_D_DECL(i,j,k),nV);
	  Uhat(KFVM_D_DECL(i,j,k),nV) = Uprev(KFVM_D_DECL(i,j,k),nV) + bhatDt*K(KFVM_D_DECL(i,j,k),nV);
	  Utmp(KFVM_D_DECL(i,j,k),nV) = Uprev(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType>
    struct RKFSAL_Stage_K {
      UViewType U,Uprev;
      KViewType Uhat,Utmp,K;
      Real delta,gam1,gam2,gam3,betaDt,bhatDt;

      RKFSAL_Stage_K(const UViewType& U_,
                     const KViewType& Uhat_,
                     const KViewType& Utmp_,
                     const UViewType& Uprev_,
                     const KViewType& K_,
                     Real delta_,
                     Real gam1_,
                     Real gam2_,
                     Real gam3_,
                     Real betaDt_,
                     Real bhatDt_):
	U(U_),
	Uprev(Uprev_),
	Uhat(Uhat_),
	Utmp(Utmp_),
	K(K_),
	delta(delta_),
	gam1(gam1_),
	gam2(gam2_),
	gam3(gam3_),
	betaDt(betaDt_),
	bhatDt(bhatDt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Utmp(KFVM_D_DECL(i,j,k),nV) += delta*U(KFVM_D_DECL(i,j,k),nV);
	  U(KFVM_D_DECL(i,j,k),nV) =
	    gam1*U(KFVM_D_DECL(i,j,k),nV) +
	    gam2*Utmp(KFVM_D_DECL(i,j,k),nV) +
	    gam3*Uprev(KFVM_D_DECL(i,j,k),nV) +
	    betaDt*K(KFVM_D_DECL(i,j,k),nV);
	  Uhat(KFVM_D_DECL(i,j,k),nV) += bhatDt*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class KViewType>
    struct RKFSAL_StageLast_K {
      KViewType Uhat,K;
      Real bhatDt;

      RKFSAL_StageLast_K(const KViewType& Uhat_,
                         const KViewType& K_,
                         Real bhatDt_):
	Uhat(Uhat_),
	K(K_),
	bhatDt(bhatDt_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Uhat(KFVM_D_DECL(i,j,k),nV) += bhatDt*K(KFVM_D_DECL(i,j,k),nV);
	}
      }
    };

    template<class UViewType,class KViewType>
    struct RKFSAL_ErrEst_K {
      UViewType U;
      KViewType Uhat;
      Real atol,rtol;

      RKFSAL_ErrEst_K(const UViewType& U_,
		  const KViewType& Uhat_,
		  Real atol_,
		  Real rtol_):
	U(U_),
	Uhat(Uhat_),
	atol(atol_),
	rtol(rtol_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k),Real& err,Real& posFlag) const {
	using Kokkos::Experimental::fabs;
	using Kokkos::Experimental::fmax;
	using Kokkos::Experimental::pow;
        using Kokkos::Experimental::isnan;

        bool hasNan = false;
	Real state[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real u = U(KFVM_D_DECL(i,j,k),nV),uhat = Uhat(KFVM_D_DECL(i,j,k),nV);
	  state[nV] = u;
          hasNan = hasNan || isnan(u);
          err += pow((u - uhat)/(atol + rtol*fmax(fabs(u),fabs(uhat))),2);
	}
	Real pos = hasNan ? -Real(1.0) : Physics::EOS<eqType>::positivity(state);
	posFlag = posFlag > pos ? pos : posFlag;
      }
    };

    template<class UViewType,class WViewType>
    struct RK_WenoSelect_K {
      UViewType U,Uprev;
      WViewType W;

      RK_WenoSelect_K(const UViewType& U_,
		      const UViewType& Uprev_,
		      const WViewType& W_):
	U(U_),
	Uprev(Uprev_),
	W(W_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const {
	using Kokkos::Experimental::fabs;
	using Kokkos::Experimental::fmin;
	using Kokkos::Experimental::fmax;
	Real u0[NUM_VARS],u1[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  u0[nV] = Uprev(KFVM_D_DECL(i,j,k),nV);
	  u1[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real e0 = Physics::EOS<eqType>::eint(u0);
	Real e1 = Physics::EOS<eqType>::eint(u1);
	W(KFVM_D_DECL(i,j,k)) = fabs(e1 - e0)/fmin(e0,e1);
      }
    };
    
  } // End numeric namespace
  
} // End KFVM namespace

#endif
