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

#include "Types.H"
#include "ProblemSetup.H"
#include "Geometry.H"
#include "Stencil.H"

namespace KFVM {

  namespace Stencil {

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
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      RS(i,j,k,FaceLabel::west  ,nQ,nR,nV) = U(i,j,k,nV) - slX/2.0;
	      RS(i,j,k,FaceLabel::east  ,nQ,nR,nV) = U(i,j,k,nV) + slX/2.0;;
	      RS(i,j,k,FaceLabel::south ,nQ,nR,nV) = U(i,j,k,nV) - slY/2.0;;
	      RS(i,j,k,FaceLabel::north ,nQ,nR,nV) = U(i,j,k,nV) + slY/2.0;;
	      RS(i,j,k,FaceLabel::bottom,nQ,nR,nV) = U(i,j,k,nV) - slZ/2.0;;
	      RS(i,j,k,FaceLabel::top   ,nQ,nR,nV) = U(i,j,k,nV) + slZ/2.0;;
	    }
	  }
	}
      }
    };
    
  } // end namespce Stencil
  
} // end namespace KFVM

#endif
