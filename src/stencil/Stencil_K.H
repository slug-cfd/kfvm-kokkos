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
      StenValsView vals;
      RSViewType RS;
      int nCells;
      StenOffsetViewType KFVM_D_DECL(lOff,tOff,ttOff);
      FaceWtViewType faceWeight;
      DerivWtViewType derivWeight;

      Weno5JS_K(const UViewType& U_,const StenValsView& vals_,const RSViewType& RS_,const int nCells_,
		KFVM_D_DECL(const StenOffsetViewType& lOff_,const StenOffsetViewType& tOff_,const StenOffsetViewType& ttOff_),
		const FaceWtViewType& face_,const DerivWtViewType& deriv_):
	U(U_),
	vals(vals_),
	RS(RS_),
	nCells(nCells_),
	KFVM_D_DECL(lOff(lOff_),tOff(tOff_),ttOff(ttOff_)),
	faceWeight(face_),
	derivWeight(deriv_) {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const int i,const int j,const int k)) const
      {
	using namespace KokkosBatched;
	
	const int nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	// Create subview slices to use gemm calls
	auto V = Kokkos::subview(vals,KFVM_D_DECL(i,j,k),Kokkos::ALL,Kokkos::ALL);
	auto RS_W = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	auto RS_E = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	auto RS_S = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	auto RS_N = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
#if (SPACE_DIM == 3)
	auto RS_B = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	auto RS_T = Kokkos::subview(RS,KFVM_D_DECL(i,j,k),int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
#endif

#ifdef KFVM_DEBUG
	std::printf("Weno5JS_K(%d,%d,%d)\n",i,j,k);
	for (int nC=0; nC<nCells; nC++) {
	  std::printf(" %2d:(lOff,tOff,ttOff) = (%d,%d,%d)\n",nC,lOff(nC),tOff(nC),ttOff(nC));
	}
#endif
	
	// Populate stencil values
	for (int nC=0; nC<nCells; nC++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    V(nC,nV) = U(KFVM_D_DECL(i + lOff(nC),j + tOff(nC),k + ttOff(nC)),nV);
	  }
	}

	// Zero out Riemann states
	for (int nQ=0; nQ<nQuad; nQ++) {
	  for (int nV=0; nV<NUM_VARS; nV++) {
            RS_W(nQ,nV) = 0.0; RS_E(nQ,nV) = 0.0;
            RS_S(nQ,nV) = 0.0; RS_N(nQ,nV) = 0.0;
#if (SPACE_DIM == 3)
            RS_B(nQ,nV) = 0.0; RS_T(nQ,nV) = 0.0;
#endif
	  }
	}

	// Fill Riemann states via gemms
	// West face
	{
	  auto rWts_W = Kokkos::subview(faceWeight,int(Stencil::SubSten::west),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_E = Kokkos::subview(faceWeight,int(Stencil::SubSten::east),
					int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_W,V,0.0,RS_W);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_W);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_E,V,1.0,RS_W);
	}
      
	// East face
	{
	  auto rWts_W = Kokkos::subview(faceWeight,int(Stencil::SubSten::west),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_E = Kokkos::subview(faceWeight,int(Stencil::SubSten::east),
					int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_W,V,0.0,RS_E);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_E);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_E,V,1.0,RS_E);
	}
	
	// South face
	{
	  auto rWts_S = Kokkos::subview(faceWeight,int(Stencil::SubSten::south),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_N = Kokkos::subview(faceWeight,int(Stencil::SubSten::north),
					int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_S,V,0.0,RS_S);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_S);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_N,V,1.0,RS_S);
	}
      
	// North face
	{
	  auto rWts_S = Kokkos::subview(faceWeight,int(Stencil::SubSten::south),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_N = Kokkos::subview(faceWeight,int(Stencil::SubSten::north),
					int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_S,V,0.0,RS_N);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_N);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_N,V,1.0,RS_N);
	}
#if (SPACE_DIM == 3)
	// Bottom face
	{
	  auto rWts_B = Kokkos::subview(faceWeight,int(Stencil::SubSten::bottom),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_T = Kokkos::subview(faceWeight,int(Stencil::SubSten::top),
					int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_B,V,0.0,RS_B);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_B);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_T,V,1.0,RS_B);
	}
      
	// Top face
	{
	  auto rWts_B = Kokkos::subview(faceWeight,int(Stencil::SubSten::bottom),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_C = Kokkos::subview(faceWeight,int(Stencil::SubSten::center),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  auto rWts_T = Kokkos::subview(faceWeight,int(Stencil::SubSten::top),
					int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.1,rWts_B,V,0.0,RS_T);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.6,rWts_C,V,1.0,RS_T);
	  SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
	    ::invoke(0.3,rWts_T,V,1.0,RS_T);
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
