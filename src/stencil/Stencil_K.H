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

#include "Types.H"
#include "ProblemSetup.H"
#include "Geometry.H"
#include "Stencil.H"

namespace KFVM {

  namespace Stencil {

    template<class UViewType,
	     class RSViewType,
	     class StenOffsetViewType,
	     class FaceWtViewType,
	     class DerivWtViewType>
    struct Weno5JS_K {
      UViewType U;
      RSViewType RS;
      StenValsView vals;
      StenIndicView indic;
      StenOffsetViewType KFVM_D_DECL(lOff,tOff,ttOff);
      FaceWtViewType faceWeight;
      DerivWtViewType derivWeight;

      Weno5JS_K(const UViewType& U_,const RSViewType& RS_,const StenValsView& vals_,const StenIndicView& indic_,
		KFVM_D_DECL(const StenOffsetViewType& lOff_,const StenOffsetViewType& tOff_,const StenOffsetViewType& ttOff_),
		const FaceWtViewType& face_,const DerivWtViewType& deriv_):
	U(U_),
	RS(RS_),
	vals(vals_),
	indic(indic_),
	KFVM_D_DECL(lOff(lOff_),tOff(tOff_),ttOff(ttOff_)),
	faceWeight(face_),
	derivWeight(deriv_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const int i,const int j,const int k)) const
      {
	using namespace KokkosBatched;
	// Linear weights
#if (SPACE_DIM == 2)
	Real gamma[] = {0.0,0.6,0.2,0.2,0.2,0.2};
#else
	Real gamma[] = {0.0,0.6,0.2,0.2,0.2,0.2,0.2,0.2};
#endif
	// Space for smoothness indicators/nonlinear weights
	Real omega[Stencil::Core::SI.nSub][NUM_VARS];
	
	// Create subview slices to use gemm calls
	auto V = Kokkos::subview(vals,KFVM_D_DECL(i,j,k),Kokkos::ALL,Kokkos::ALL);
	auto beta = Kokkos::subview(indic,KFVM_D_DECL(i,j,k),Kokkos::ALL,Kokkos::ALL);
	auto RS_W = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	auto RS_E = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	auto RS_S = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	auto RS_N = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
#if (SPACE_DIM == 3)
	auto RS_B = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	auto RS_T = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
#endif
	
	// Populate stencil values
	for (int nC=0; nC<Stencil::Core::SI.nCellsFull; nC++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    V(nC,nV) = U(KFVM_D_DECL(i + lOff(nC),j + tOff(nC),k + ttOff(nC)),nV);
	  }
	}

	// Zero out Riemann states
	for (int nQ=0; nQ<Stencil::Core::SI.nqFace_d; nQ++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
            RS_W(nQ,nV) = 0.0; RS_E(nQ,nV) = 0.0;
            RS_S(nQ,nV) = 0.0; RS_N(nQ,nV) = 0.0;
#if (SPACE_DIM == 3)
            RS_B(nQ,nV) = 0.0; RS_T(nQ,nV) = 0.0;
#endif
	  }
	}

	// Calculate smoothness indicators
	for (int nS=0; nS<Stencil::Core::SI.nSub; nS++) {
	  // use gemm to get all derivatives of all components together
	  auto dWts = Kokkos::subview(derivWeight,nS,Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(1.0,dWts,V,0.0,beta);
	  
	  // Accumulate derivatives into single indicator
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    omega[nS][nV] = 0.0;
	    for (int nI=0; nI<Stencil::Core::SI.nIndic; nI++) {
	      omega[nS][nV] += beta(nI,nV)*beta(nI,nV);
	    }
	  }
	}

	// Calculate nonlinear weights
	for (int nS=0; nS<Stencil::Core::SI.nSub; nS++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    omega[nS][nV] = gamma[nS]/pow(1.e-8 + omega[nS][nV],2);
	  }
	}

	// Normalize in hacky way
	// Using a multiD weno pattern for dim-by-dim is odd
	Real ctrWts[] = {omega[Stencil::SubSten::center][0],
			 omega[Stencil::SubSten::center][1],
			 omega[Stencil::SubSten::center][2],
			 omega[Stencil::SubSten::center][3],
			 omega[Stencil::SubSten::center][4]};
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real tot = omega[Stencil::SubSten::west][nV] + omega[Stencil::SubSten::center][nV] + omega[Stencil::SubSten::east][nV];
	  omega[Stencil::SubSten::west][nV] /= tot;
	  omega[Stencil::SubSten::center][nV] /= tot;
	  omega[Stencil::SubSten::east][nV] /= tot;
	}

	// Fill Riemann states using gemv component by component
	// West face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_W = Kokkos::subview(faceWeight,int(Stencil::SubSten::west),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_E = Kokkos::subview(faceWeight,int(Stencil::SubSten::east),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_W,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::west][nV],rWts_W,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::east][nV],rWts_E,in,1.0,out);
	}
      
	// East face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_W = Kokkos::subview(faceWeight,int(Stencil::SubSten::west),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_E = Kokkos::subview(faceWeight,int(Stencil::SubSten::east),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_E,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::west][nV],rWts_W,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::east][nV],rWts_E,in,1.0,out);
	}

	// Normalize in hacky way
	// Using a multiD weno pattern for dim-by-dim is odd
	omega[Stencil::SubSten::center][0] = ctrWts[0];
	omega[Stencil::SubSten::center][1] = ctrWts[1];
	omega[Stencil::SubSten::center][2] = ctrWts[2];
	omega[Stencil::SubSten::center][3] = ctrWts[3];
	omega[Stencil::SubSten::center][4] = ctrWts[4];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real tot = omega[Stencil::SubSten::south][nV] + omega[Stencil::SubSten::center][nV] + omega[Stencil::SubSten::north][nV];
	  omega[Stencil::SubSten::south][nV] /= tot;
	  omega[Stencil::SubSten::center][nV] /= tot;
	  omega[Stencil::SubSten::north][nV] /= tot;
	}
	
	// South face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_S = Kokkos::subview(faceWeight,int(Stencil::SubSten::south),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_N = Kokkos::subview(faceWeight,int(Stencil::SubSten::north),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_S,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::south][nV],rWts_S,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::north][nV],rWts_N,in,1.0,out);
	}
      
	// North face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_S = Kokkos::subview(faceWeight,int(Stencil::SubSten::south),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_N = Kokkos::subview(faceWeight,int(Stencil::SubSten::north),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_N,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::south][nV],rWts_S,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::north][nV],rWts_N,in,1.0,out);
	}
#if (SPACE_DIM == 3)
	// Normalize in hacky way
	// Using a multiD weno pattern for dim-by-dim is odd
	omega[Stencil::SubSten::center][0] = ctrWts[0];
	omega[Stencil::SubSten::center][1] = ctrWts[1];
	omega[Stencil::SubSten::center][2] = ctrWts[2];
	omega[Stencil::SubSten::center][3] = ctrWts[3];
	omega[Stencil::SubSten::center][4] = ctrWts[4];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real tot = omega[Stencil::SubSten::bottom][nV] + omega[Stencil::SubSten::center][nV] + omega[Stencil::SubSten::top][nV];
	  omega[Stencil::SubSten::bottom][nV] /= tot;
	  omega[Stencil::SubSten::center][nV] /= tot;
	  omega[Stencil::SubSten::top][nV] /= tot;
	}
	
	// Bottom face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_B = Kokkos::subview(faceWeight,int(Stencil::SubSten::bottom),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_T = Kokkos::subview(faceWeight,int(Stencil::SubSten::top),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_B,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::bottom][nV],rWts_B,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::top][nV],rWts_T,in,1.0,out);
	}
      
	// Top face
	for (int nV=0; nV<NUM_VARS; nV++) {
	  auto rWts_B = Kokkos::subview(faceWeight,int(Stencil::SubSten::bottom),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_T = Kokkos::subview(faceWeight,int(Stencil::SubSten::top),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  auto in = Kokkos::subview(V,Kokkos::ALL,nV);
	  auto out = Kokkos::subview(RS_T,Kokkos::ALL,nV);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::bottom][nV],rWts_B,in,0.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::center][nV],rWts_C,in,1.0,out);
	  SerialGemv<Trans::NoTranspose,Algo::Gemv::Unblocked>::invoke(omega[Stencil::SubSten::top][nV],rWts_T,in,1.0,out);
	}
#endif
      }
    };

    template<class UViewType,class RSViewType>
    struct MinModRecon_K {
      UViewType U;
      RSViewType RS;

      MinModRecon_K(const UViewType& U_,const RSViewType& RS_): U(U_),RS(RS_) {}

      KOKKOS_INLINE_FUNCTION
      Real minmod(Real ul,Real uc,Real ur) const {
	using Kokkos::Experimental::fabs;
	return fabs(ur - uc)<fabs(uc - ul) ? (ur - uc) : (uc - ul);
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real slX = minmod(U(i-1,j,nV),U(i,j,nV),U(i+1,j,nV));
	  Real slY = minmod(U(i,j-1,nV),U(i,j,nV),U(i,j+1,nV));
	  // Loop over quadrature points
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    RS(i,j,FaceLabel::west ,nQ,nV) = U(i,j,nV) - slX/2.0;
	    RS(i,j,FaceLabel::east ,nQ,nV) = U(i,j,nV) + slX/2.0;
	    RS(i,j,FaceLabel::south,nQ,nV) = U(i,j,nV) - slY/2.0;
	    RS(i,j,FaceLabel::north,nQ,nV) = U(i,j,nV) + slY/2.0;
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,const int k) const {
	// Loop over SimVars
	for (int nV=0; nV<NUM_VARS; nV++) {
	  Real slX = minmod(U(i-1,j,k,nV),U(i,j,k,nV),U(i+1,j,k,nV));
	  Real slY = minmod(U(i,j-1,k,nV),U(i,j,k,nV),U(i,j+1,k,nV));
	  Real slZ = minmod(U(i,j,k-1,nV),U(i,j,k,nV),U(i,j,k+1,nV));
	  // Loop over quadrature points
	  for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	    RS(i,j,k,FaceLabel::west  ,nQ,nV) = U(i,j,k,nV) - slX/2.0;
	    RS(i,j,k,FaceLabel::east  ,nQ,nV) = U(i,j,k,nV) + slX/2.0;
	    RS(i,j,k,FaceLabel::south ,nQ,nV) = U(i,j,k,nV) - slY/2.0;
	    RS(i,j,k,FaceLabel::north ,nQ,nV) = U(i,j,k,nV) + slY/2.0;
	    RS(i,j,k,FaceLabel::bottom,nQ,nV) = U(i,j,k,nV) - slZ/2.0;
	    RS(i,j,k,FaceLabel::top   ,nQ,nV) = U(i,j,k,nV) + slZ/2.0;
	  }
	}
      }
    };
    
  } // end namespce Stencil
  
} // end namespace KFVM

#endif
