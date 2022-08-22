#ifndef STENCIL_K_H_
#define STENCIL_K_H_

// File: Stencil_K.H
// Purpose: Define compute kernels for reconstructing Riemann states
//          from cell average data

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <Kokkos_View.hpp>

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>

#include "../Types.H"
#include "../ProblemSetup.H"
#include "../Geometry.H"
#include "../hydro/Hydro_K.H"
#include "Stencil.H"

namespace KFVM {

  namespace Stencil {

    template<class UViewType>
    struct KernelWenoRecon_K {
      UViewType U;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      WorkView Work;
      OffsetView KFVM_D_DECL(lOff,tOff,ttOff);
      SubIdxView subIdx;
      FaceWtView faceWeight;
      CellWtView derivWeight;

      KernelWenoRecon_K(const UViewType& U_,
			KFVM_D_DECL(const FaceDataView& rsX_,
				    const FaceDataView& rsY_,
				    const FaceDataView& rsZ_),
			const WorkView& Work_,
			KFVM_D_DECL(const OffsetView& lOff_,
				    const OffsetView& tOff_,
				    const OffsetView& ttOff_),
			const SubIdxView& subIdx_,
			const FaceWtView& face_,
			const CellWtView& deriv_):
	U(U_),
	KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
	Work(Work_),
	KFVM_D_DECL(lOff(lOff_),tOff(tOff_),ttOff(ttOff_)),
	subIdx(subIdx_),
	faceWeight(face_),
	derivWeight(deriv_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const int i,const int j,const int k)) const
      {
	using namespace KokkosBatched;
	using Kokkos::Experimental::fabs;
	using Kokkos::Experimental::pow;
	constexpr int nCellsFull = Stencil::Core::SI.nCellsFull;
	constexpr int nCellsCtr = Stencil::Core::SI.nCellsCtr;
	constexpr int nCellsBias = Stencil::Core::SI.nCellsBias;
	constexpr int nIndic = Stencil::Core::SI.nIndic;

	// Linear weights and epsilon value
	constexpr Real eps = Real(1.e-8);
#if (SPACE_DIM == 2)
	constexpr Real gHi = 0.8,gLo = 0.8;
	Real gamma[] = {gHi,
			(1.0 - gHi)*gLo,
			(1.0 - gHi)*(1.0 - gLo)/4.0,
			(1.0 - gHi)*(1.0 - gLo)/4.0,
			(1.0 - gHi)*(1.0 - gLo)/4.0,
			(1.0 - gHi)*(1.0 - gLo)/4.0};
#else
	constexpr Real gHi = 0.75,gLo = 0.75;
	Real gamma[] = {gHi,
			(1.0 - gHi)*gLo,
			(1.0 - gHi)*(1.0 - gLo)/6.0,
			(1.0 - gHi)*(1.0 - gLo)/6.0,
			(1.0 - gHi)*(1.0 - gLo)/6.0,
			(1.0 - gHi)*(1.0 - gLo)/6.0,
			(1.0 - gHi)*(1.0 - gLo)/6.0,
			(1.0 - gHi)*(1.0 - gLo)/6.0};
#endif
	
	// Create subview slices for values on each (sub)stencil
	auto Vf = Kokkos::subview(Work,KFVM_D_DECL(i,j,k),
				  Kokkos::make_pair(0,nCellsFull),Kokkos::ALL);
	auto Vc = Kokkos::subview(Work,KFVM_D_DECL(i,j,k),
				  Kokkos::make_pair(0,nCellsCtr),Kokkos::ALL);
	auto Vb = Kokkos::subview(Work,KFVM_D_DECL(i,j,k),
				  Kokkos::make_pair(nCellsFull,
						     nCellsFull+nCellsBias),Kokkos::ALL);
	// Subview slice to store derivatives in before accumulating into beta
	auto B = Kokkos::subview(Work,KFVM_D_DECL(i,j,k),
				 Kokkos::make_pair(nCellsFull+nCellsBias,
						    nCellsFull+nCellsBias+nIndic),Kokkos::ALL);

	// Storage for nonlinear weights
	Real omega[Stencil::Core::SI.nSub][NUM_VARS];

	// Storage for reference state
	Real refState[NUM_VARS];
	
	// Populate stencil values
	for (int nC=0; nC<Stencil::Core::SI.nCellsFull; nC++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    Vf(nC,nV) = U(KFVM_D_DECL(i + lOff(nC),
				      j + tOff(nC),
				      k + ttOff(nC)),nV);
	  }
	}

	// Save reference state and convert to primitive vars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	
	for (int nC=0; nC<Stencil::Core::SI.nCellsFull; nC++) {
	  auto state = Kokkos::subview(Vf,nC,Kokkos::ALL);
	  Hydro::Cons2Prim(state,refState,1.4);
	}

	// Find all smoothness indicators
	// For unknown reasons this gemm call can not be moved into a helper function
	// Full stencil
	{
	  auto dWts = Kokkos::subview(derivWeight,0,Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>::invoke(Real(1.0),dWts,Vf,Real(0.0),B);
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    omega[0][nV] = 0.0;
	    for (int nI=0; nI<Stencil::Core::SI.nIndic; nI++) {
	      omega[0][nV] += B(nI,nV)*B(nI,nV);
	    }
	  }
	}
	// Central substencil
	{
	  auto dWts = Kokkos::subview(derivWeight,1,Kokkos::ALL,Kokkos::make_pair(0,nCellsCtr));
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>::invoke(Real(1.0),dWts,Vc,Real(0.0),B);
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    omega[1][nV] = 0.0;
	    for (int nI=0; nI<Stencil::Core::SI.nIndic; nI++) {
	      omega[1][nV] += B(nI,nV)*B(nI,nV);
	    }
	  }
	}
	// biased substencils
	for (int nB=0; nB<Stencil::Core::SI.nBias; nB++) {
	  fillBiased<decltype(Vf),decltype(Vb)>(nB,Vf,Vb);
	  auto dWts = Kokkos::subview(derivWeight,nB + 2,Kokkos::ALL,Kokkos::make_pair(0,nCellsBias));
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>::invoke(Real(1.0),dWts,Vb,Real(0.0),B);
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    omega[nB + 2][nV] = 0.0;
	    for (int nI=0; nI<Stencil::Core::SI.nIndic; nI++) {
	      omega[nB + 2][nV] += B(nI,nV)*B(nI,nV);
	    }
	  }
	}

	// Convert smoothness indicators to omegas
	for (int nV=0; nV<NUM_VARS; nV++) {
#if (SPACE_DIM == 2)
	  Real tau = fabs(omega[0][nV] - (omega[2][nV] + omega[3][nV] +
					  omega[4][nV] + omega[5][nV])/4.0);
#else
	  Real tau = fabs(omega[0][nV] - (omega[2][nV] + omega[3][nV] +
					  omega[4][nV] + omega[5][nV] +
					  omega[6][nV] + omega[7][nV])/6.0);
#endif
	  Real tot = 0.0;
	  for (int nS=0; nS<Stencil::Core::SI.nSub; nS++) {
	    omega[nS][nV] = gamma[nS]*(1.0 + pow(tau/(omega[nS][nV] + eps),STENCIL_RADIUS));
	    tot += omega[nS][nV];
	  }
	  // normalize them
	  for (int nS=0; nS<Stencil::Core::SI.nSub; nS++) {
	    omega[nS][nV] /= tot;
	  }
	}

	// convert omegas to Weno-AO style weights
	for (int nV=0; nV<NUM_VARS; nV++) {
	  for (int nS=1; nS<Stencil::Core::SI.nSub; nS++) {
	    omega[nS][nV] -= omega[0][nV]*gamma[nS]/gamma[0];
	  }
	  omega[0][nV] /= gamma[0];
	}

	// Pull Riemann states subviews out into array
#if (SPACE_DIM == 2)
	decltype(Kokkos::subview(rsX,i,j,1,Kokkos::ALL,Kokkos::ALL)) rsFace[] = {
	  Kokkos::subview(rsX,i    ,j,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsX,i + 1,j,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j    ,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j + 1,0,Kokkos::ALL,Kokkos::ALL)
	};
#else
	decltype(Kokkos::subview(rsX,i,j,k,1,Kokkos::ALL,Kokkos::ALL)) rsFace[] = {
	  Kokkos::subview(rsX,i    ,j,k,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsX,i + 1,j,k,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j    ,k,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j + 1,k,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsZ,i,j,k    ,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsZ,i,j,k + 1,0,Kokkos::ALL,Kokkos::ALL)
	};
#endif
	
	// Fill Riemann states using gemv one face at a time
	for (int nF=0; nF<2*SPACE_DIM; nF++) {
	  auto rWtsF = Kokkos::subview(faceWeight,0,nF,Kokkos::ALL,Kokkos::ALL);
	  auto rWtsC = Kokkos::subview(faceWeight,1,nF,Kokkos::ALL,Kokkos::make_pair(0,nCellsCtr));
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    auto rsFV = Kokkos::subview(rsFace[nF],Kokkos::ALL,nV);
	    auto vf = Kokkos::subview(Vf,Kokkos::ALL,nV);
	    auto vc = Kokkos::subview(Vc,Kokkos::ALL,nV);
	    SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[0][nV],rWtsF,vf,Real(0.0),rsFV);
	    SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[1][nV],rWtsC,vc,Real(1.0),rsFV);
	  }
	}
	
	// Biased substencils
	for (int nB=0; nB<Stencil::Core::SI.nBias; nB++) {
	  fillBiased<decltype(Vf),decltype(Vb)>(nB,Vf,Vb);
	  for (int nF=0; nF<2*SPACE_DIM; nF++) {
	    auto rWtsB = Kokkos::subview(faceWeight,2 + nB,nF,Kokkos::ALL,Kokkos::make_pair(0,nCellsBias));
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      auto rsFV = Kokkos::subview(rsFace[nF],Kokkos::ALL,nV);
	      auto vb = Kokkos::subview(Vb,Kokkos::ALL,nV);
	      SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[2 + nB][nV],rWtsB,vb,Real(1.0),rsFV);
	    }
	  }
	}

	// Convert back to conservative variables
	for (int nF=0; nF<2*SPACE_DIM; nF++) {
	  for (int nQ=0; nQ<Stencil::Core::SI.nqFace_d; nQ++) {
	    auto state = Kokkos::subview(rsFace[nF],nQ,Kokkos::ALL);
	    Hydro::Prim2Cons(state,refState,1.4);
	  }
	}
      }
      
      // put all helper functions inside private
    private:
      template<class ValsFullType,class ValsBiasType>
      KOKKOS_INLINE_FUNCTION
      void fillBiased(int nB,ValsFullType Vf,ValsBiasType Vb) const
      {
	for (int nV=0; nV<NUM_VARS; nV++) {
	  for (int nC=0; nC<Stencil::Core::SI.nCellsBias; nC++) {
	    Vb(nC,nV) = Vf(subIdx(nB,nC),nV);
	  }
	}
      }
    };
    
    template<class UViewType>
    struct KernelLinearRecon_K {
      UViewType U;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      WorkView Work;
      OffsetView KFVM_D_DECL(lOff,tOff,ttOff);
      FaceWtView faceWeight;

      KernelLinearRecon_K(const UViewType& U_,
			  KFVM_D_DECL(const FaceDataView& rsX_,
				      const FaceDataView& rsY_,
				      const FaceDataView& rsZ_),
			  const WorkView& Work_,
			  KFVM_D_DECL(const OffsetView& lOff_,
				      const OffsetView& tOff_,
				      const OffsetView& ttOff_),
			  const FaceWtView& face_):
	U(U_),
	KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
	Work(Work_),
	KFVM_D_DECL(lOff(lOff_),tOff(tOff_),ttOff(ttOff_)),
	faceWeight(face_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const int i,const int j,const int k)) const
      {
	using namespace KokkosBatched;
		
	// Create subview slices to use gemm calls
	auto V = Kokkos::subview(Work,KFVM_D_DECL(i,j,k),
				 Kokkos::make_pair(0,Stencil::Core::SI.nCellsFull),
				 Kokkos::ALL);
	
	// Populate stencil values
	for (int nC=0; nC<Stencil::Core::SI.nCellsFull; nC++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    V(nC,nV) = U(KFVM_D_DECL(i + lOff(nC),
				     j + tOff(nC),
				     k + ttOff(nC)),nV);
	  }
	}

	// Pull Riemann states subviews out into array
#if (SPACE_DIM == 2)
	decltype(Kokkos::subview(rsX,i,j,1,Kokkos::ALL,Kokkos::ALL)) rsFace[] = {
	  Kokkos::subview(rsX,i    ,j,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsX,i + 1,j,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j    ,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j + 1,0,Kokkos::ALL,Kokkos::ALL)
	};
#else
	decltype(Kokkos::subview(rsX,i,j,k,1,Kokkos::ALL,Kokkos::ALL)) rsFace[] = {
	  Kokkos::subview(rsX,i    ,j,k,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsX,i + 1,j,k,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j    ,k,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsY,i,j + 1,k,0,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsZ,i,j,k    ,1,Kokkos::ALL,Kokkos::ALL),
	  Kokkos::subview(rsZ,i,j,k + 1,0,Kokkos::ALL,Kokkos::ALL)
	};
#endif

	// Fill Riemann states using gemm one face at a time
	for (int nF=0; nF<2*SPACE_DIM; nF++) {
	  auto rWts = Kokkos::subview(faceWeight,int(Stencil::SubSten::full),
				      nF,Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(1.0,rWts,V,0.0,rsFace[nF]);
	}
      }
    };

    template<class UViewType>
    struct MinModRecon_K {
      UViewType U;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);

      MinModRecon_K(const UViewType& U_,
		    KFVM_D_DECL(const FaceDataView& rsX_,
				const FaceDataView& rsY_,
				const FaceDataView& rsZ_)):
	U(U_),
	KFVM_D_DECL(rsX(rsX_),
		    rsY(rsY_),
		    rsZ(rsZ_))
      {}

      KOKKOS_INLINE_FUNCTION
      Real minmod(Real ul,Real uc,Real ur) const {
	using Kokkos::Experimental::fabs;
	return fabs(ur - uc)<fabs(uc - ul) ? (ur - uc) : (uc - ul);
      }
#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real slX = minmod(U(i-1,j,nV),U(i,j,nV),U(i+1,j,nV));
	  Real slY = minmod(U(i,j-1,nV),U(i,j,nV),U(i,j+1,nV));
	  // Loop over quadrature points
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    rsX(i,j,1,nQ,nV) = U(i,j,nV) - slX/2.0;
	    rsX(i + 1,j,0,nQ,nV) = U(i,j,nV) + slX/2.0;
	    rsY(i,j,1,nQ,nV) = U(i,j,nV) - slY/2.0;
	    rsY(i,j + 1,0,nQ,nV) = U(i,j,nV) + slY/2.0;
	  }
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,const int k) const {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real slX = minmod(U(i-1,j,k,nV),U(i,j,k,nV),U(i+1,j,k,nV));
	  Real slY = minmod(U(i,j-1,k,nV),U(i,j,k,nV),U(i,j+1,k,nV));
	  Real slZ = minmod(U(i,j,k-1,nV),U(i,j,k,nV),U(i,j,k+1,nV));
	  // Loop over quadrature points
	  for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	    rsX(i,j,k,1,nQ,nV) = U(i,j,k,nV) - slX/2.0;
	    rsX(i + 1,j,k,0,nQ,nV) = U(i,j,k,nV) + slX/2.0;
	    rsY(i,j,k,1,nQ,nV) = U(i,j,k,nV) - slY/2.0;
	    rsY(i,j + 1,k,0,nQ,nV) = U(i,j,k,nV) + slY/2.0;
	    rsZ(i,j,k,1,nQ,nV) = U(i,j,k,nV) - slZ/2.0;
	    rsZ(i,j,k + 1,0,nQ,nV) = U(i,j,k,nV) + slZ/2.0;
	  }
	}
      }
#endif
    };
    
  } // end namespce Stencil
  
} // end namespace KFVM

#endif
