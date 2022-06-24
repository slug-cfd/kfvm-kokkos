#ifndef NUMERIC_K_H_
#define NUMERIC_K_H_

// File: Numeric_K.H
// Purpose: Define compute kernels for various simple numerical methods
//          referencing data from Numeric.H

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_View.hpp>
#include <cstdint>
#include <cstdio>

#include "../Types.H"
#include "../ProblemSetup.H"
#include "../Geometry.H"
#include "Numeric.H"

namespace KFVM {

  namespace {
    KOKKOS_INLINE_FUNCTION
    void InitialCondition(Real x,Real y,Real z,Real state[NUM_VARS])
    {
      (void) y;
      (void) z;
      state[SimVar::dens] = x<0.5 ? 1.0 : 0.125;
      state[SimVar::momx] = 0.0;
      state[SimVar::momy] = 0.0;
      state[SimVar::momz] = 0.0;
      state[SimVar::etot] = (x<0.5 ? 1.0 : 0.1)/0.4;
    }
  }

  namespace Numeric {

    template<class DataViewType,class QRViewType,int nQuad>
    struct IntegrateIC_K
    {
      DataViewType v;
      QRViewType ab,wt;
      Geometry geom;

      IntegrateIC_K(const DataViewType& v_,
		    const QRViewType& ab_,
		    const QRViewType& wt_,
		    const Geometry& geom_):
	v(v_),
	ab(ab_),
	wt(wt_),
	geom(geom_) {}

      // Helper function to convert abscissae on (-0.5,0.5)
      // to the bounds of a cell
      KOKKOS_INLINE_FUNCTION
      Real sclAb(const Real ctr,const Real del,const Real ab) const
      {
	return ab*del + ctr;
      }

      // 2D version
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j) const {	
	// Center of cell (i,j)
	Real xCtr = geom.xCoord(i),yCtr = geom.yCoord(j);

	// Zero out state
	for (int nV=0; nV<NUM_VARS; nV++) {
	  v(i,j,nV) = 0.0;
	}

	// Loop over quadrature points and accumulate integral
	for (int nX=0; nX<nQuad; nX++) {
	  for (int nY=0; nY<nQuad; nY++) {
	    // Quadrature point
	    Real abX = sclAb(xCtr,geom.dx,ab(nX));
	    Real abY = sclAb(yCtr,geom.dy,ab(nY));

	    // IC at quadrature point
	    Real state[NUM_VARS];
	    InitialCondition(abX,abY,geom.zLo,state);

	    // Add each component into IC
	    for (int nV=0; nV<NUM_VARS; nV++) {
	      v(i,j,nV) += wt(nX)*wt(nY)*state[nV];
	    }
	  }
	}
      }

      // 3D version
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,const int k) const {
	// Center of cell (i,j,k)
	Real xCtr = geom.xCoord(i),yCtr = geom.yCoord(j),zCtr = geom.zCoord(k);

	// Zero out state
	for (int nV=0; nV<NUM_VARS; nV++) {
	  v(i,j,k,nV) = 0.0;
	}

	// Loop over quadrature points and accumulate integral
	for (int nX=0; nX<nQuad; nX++) {
	  for (int nY=0; nY<nQuad; nY++) {
	    for (int nZ=0; nZ<nQuad; nZ++) {
	      // Quadrature point
	      Real abX = sclAb(xCtr,geom.dx,ab(nX));
	      Real abY = sclAb(yCtr,geom.dy,ab(nY));
	      Real abZ = sclAb(zCtr,geom.dz,ab(nZ));

	      // IC at quadrature point
	      Real state[NUM_VARS];
	      InitialCondition(abX,abY,abZ,state);

	      // Add each component into IC
	      for (int nV=0; nV<NUM_VARS; nV++) {
		v(i,j,k,nV) += wt(nX)*wt(nY)*wt(nZ)*state[nV];
	      }
	    }
	  }
	}
      }
      
    };

  } // End numeric namespace
  
} // End KFVM namespace

#endif
