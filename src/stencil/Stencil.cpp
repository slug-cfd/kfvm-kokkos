#include "SimVar.H"
#include "Types.H"
#include "Stencil.H"
#include <cstdio>

namespace KFVM {

  namespace Stencil {

    int Stencil::Core::coord2idx(int i,int j)
    {
      int idx = -1;
      for (int n=0; n<Stencil::Core::SI.nCellsFull; n++) {
	if (i == lOff[n] && j == tOff[n]) {
	  idx = n;
	  break;
	}
      }
      return idx;
    }
    
    int Stencil::Core::coord2idx(int i,int j,int k)
    {
      int idx = -1;
      for (int n=0; n<Stencil::Core::SI.nCellsFull; n++) {
	if (i == lOff[n] && j == tOff[n] && k == ttOff[n]) {
	  idx = n;
	  break;
	}
      }
      return idx;
    }
    
    void Stencil::Core::findGammas()
    {
      Real gHi = SPACE_DIM==2 ? 0.8 : 0.7;
      Real gLo = gHi;
      // Set just the CWENO weights of the low order scheme
      gamma[1] = gLo;
      for (int nS=2; nS<SI.nSub; nS++) {
        gamma[nS] = (1.0 - gLo)/Real(SI.nBias);
      }
      // Rescale for large central stencil
      gamma[0] = gHi;
      for (int nS=1; nS<SI.nSub; nS++) {
        gamma[nS] *= (1.0 - gHi);
      }
    }
    
    void Stencil::Core::findStencil()
    {
#if (SPACE_DIM == 2)
      double mjRsq = rad*rad;
      int nC = 1;
      // Assure that central substencil is first
      lOff[0] = 0;
      tOff[0] = 0;
      for (int nL=1; nL<=rad; nL++) {
        // On axis always fits
	lOff[nC] = -nL; tOff[nC] = 0;   nC++;
        lOff[nC] = nL;  tOff[nC] = 0;   nC++;
        lOff[nC] = 0;   tOff[nC] = -nL; nC++;
        lOff[nC] = 0;   tOff[nC] = nL;  nC++;
        // Check if diagonal cells fit in radius
	for (int nR=1; nR<=nL; nR++) {
	  double rsq = 2.0*nR*nR;
	  int lyr = static_cast<int>(ceil(sqrt(rsq)));
	  if (rsq <= mjRsq && lyr == nL) {
	    lOff[nC] = -nR; tOff[nC] = -nR; nC++;
	    lOff[nC] = -nR; tOff[nC] = nR;  nC++;
	    lOff[nC] = nR;  tOff[nC] = -nR; nC++;
	    lOff[nC] = nR;  tOff[nC] = nR;  nC++;
	  }
	}
        // Look at remaining cells to fill circle
	for (int i=2; i<=rad; i++) {
	  for (int j=1; j<i; j++) {
	    double cx = i,cy = j,rsq = cx*cx + cy*cy;
	    int lyr = static_cast<int>(ceil(sqrt(rsq)));
	    if (rsq <= mjRsq && lyr == nL) {
	      lOff[nC] = -i; tOff[nC] = j;  nC++;
	      lOff[nC] = -i; tOff[nC] = -j; nC++;
	      lOff[nC] = i;  tOff[nC] = j;  nC++;
	      lOff[nC] = i;  tOff[nC] = -j; nC++;
	      
	      lOff[nC] = -j; tOff[nC] = i;  nC++;
	      lOff[nC] = -j; tOff[nC] = -i; nC++;
	      lOff[nC] = j;  tOff[nC] = i;  nC++;
	      lOff[nC] = j;  tOff[nC] = -i; nC++;
	    }
	  }
	}
      }
#else
      int mjRsq = rad*rad;
      int rm = rad - 1,nC = 0;
      // Assure that central substencil is first
      for (int i=-rm; i<=rm; i++) {
	for (int j=-rm; j<=rm; j++) {
	  for (int k=-rm; k<=rm; k++) {
	    if ((i*i + j*j + k*k) <= rm*rm) {
	      lOff[nC] = i; tOff[nC] = j; ttOff[nC] = k; nC++;
	    }
	  }
	}
      }

      // then add all other cells
      for (int i=-rad; i<=rad; i++) {
	for (int j=-rad; j<=rad; j++) {
	  for (int k=-rad; k<=rad; k++) {
	    int rsq = i*i + j*j + k*k;
	    if (rsq > rm*rm && rsq <= mjRsq) {
	      lOff[nC] = i; tOff[nC] = j; ttOff[nC] = k; nC++;
	    }
	  }
	}
      }
#endif
      // Find biased substencils
      findSubStencils();
    }

    void Stencil::Core::findSubStencils()
    {
      // Assign substencil indices that map into full stencil
#if (SPACE_DIM == 2)
      int nN = 0,nS = 0,nE = 0,nW = 0;
      for (int h=0; h<=rad; h++) {
        // Set on-axis indices first
        subIdx[                  nE++] = coord2idx( h, 0);
        subIdx[  SI.nCellsBias + nW++] = coord2idx(-h, 0);
        subIdx[2*SI.nCellsBias + nN++] = coord2idx( 0, h);
        subIdx[3*SI.nCellsBias + nS++] = coord2idx( 0,-h);
      }
      // Set off axis indices in pairs 
      for (int h=0; h<=rad; h++) {
        for (int l=1; l<=h; l++) {
	  int idx  = coord2idx(h,-l);
	  int idxp = coord2idx(h, l);
          if (idx >= 0) {
            subIdx[nE++] = idx;
            subIdx[nE++] = idxp;
        
            idx  = coord2idx(-h,-l);
            idxp = coord2idx(-h, l);
            subIdx[SI.nCellsBias + nW++] = idx;
            subIdx[SI.nCellsBias + nW++] = idxp;
	    
	    idx  = coord2idx(-l,h);
	    idxp = coord2idx( l,h);
            subIdx[2*SI.nCellsBias + nN++] = idx;
            subIdx[2*SI.nCellsBias + nN++] = idxp;
        
            idx  = coord2idx(-l,-h);
            idxp = coord2idx( l,-h);
            subIdx[3*SI.nCellsBias + nS++] = idx;
            subIdx[3*SI.nCellsBias + nS++] = idxp;
          }
        }
      }
#else
      int nN = 0,nS = 0,nE = 0,nW = 0,nT = 0,nB = 0;
      for (int n=0; n<SI.nCellsFull; n++) {
	// check west substencil
	if (-lOff[n] > abs(tOff[n]) && -lOff[n] > abs(ttOff[n])) {
	  subIdx[nW++] = n;
	}
	
	// check east substencil
	if (lOff[n] > abs(tOff[n]) && lOff[n] > abs(ttOff[n])) {
	  subIdx[SI.nCellsBias + nE++] = n;
	}
	
	// check south substencil
	if (-tOff[n] > abs(lOff[n]) && -tOff[n] > abs(ttOff[n])) {
	  subIdx[2*SI.nCellsBias + nS++] = n;
	}
	
	// check north substencil
	if (tOff[n] > abs(lOff[n]) && tOff[n] > abs(ttOff[n])) {
	  subIdx[3*SI.nCellsBias + nN++] = n;
	}
	
	// check bottom substencil
	if (-ttOff[n] > abs(lOff[n]) && -ttOff[n] > abs(tOff[n])) {
	  subIdx[4*SI.nCellsBias + nB++] = n;
	}
	
	// check top substencil
	if (ttOff[n] > abs(lOff[n]) && ttOff[n] > abs(tOff[n])) {
	  subIdx[5*SI.nCellsBias + nT++] = n;
	}
      }
#endif
    }

    void Stencil::findWeights(double lfac,const QuadRuleView& ab)
    {
      // Make host accessible views
      auto h_face = Kokkos::create_mirror(faceWeights);
      auto h_deriv = Kokkos::create_mirror(derivWeights);
      
      // Fill with zeros initially
      for (int nS=0; nS<core.SI.nSub; nS++) {
	for (int nC=0; nC<core.SI.nCellsFull; nC++) {
	  for (int nD=0; nD<2*SPACE_DIM; nD++) {
	    for (int nQ=0; nQ<core.SI.nqFace_d; nQ++) {
	      h_face(nS,nD,nQ,nC) = 0.0;
	    }
	  }
	  
	  for (int nQ=0; nQ<core.SI.nDeriv*core.SI.nqCell_d; nQ++) {
	    h_deriv(nS,nQ,nC) = 0.0;
	  }
	}
      }

      // Fill in face weights
#if (SPACE_DIM == 2)
      for (int nQ=0; nQ<core.SI.nqFace_d; nQ++) {
	const int ctr = 0;
	const int xm1 = 1,xp1 = 2,xm2 = 5,xp2 = 6;
	const int ym1 = 3,yp1 = 4,ym2 = 7,yp2 = 8;
	// Reconstruction weights for east face
	h_face(SubSten::west,FaceLabel::east,nQ,xm2) =  1.0/3.0;
	h_face(SubSten::west,FaceLabel::east,nQ,xm1) = -7.0/6.0;
	h_face(SubSten::west,FaceLabel::east,nQ,ctr) = 11.0/6.0;
	
	h_face(SubSten::center,FaceLabel::east,nQ,xm1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::east,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::east,nQ,xp1) =  1.0/3.0;
	
	h_face(SubSten::east,FaceLabel::east,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::east,FaceLabel::east,nQ,xp1) =  5.0/6.0;
	h_face(SubSten::east,FaceLabel::east,nQ,xp2) = -1.0/6.0;
	
	// Reconstruction weights for north face
	h_face(SubSten::south,FaceLabel::north,nQ,ym2) =  1.0/3.0;
	h_face(SubSten::south,FaceLabel::north,nQ,ym1) = -7.0/6.0;
	h_face(SubSten::south,FaceLabel::north,nQ,ctr) = 11.0/6.0;
	
	h_face(SubSten::center,FaceLabel::north,nQ,ym1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::north,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::north,nQ,yp1) =  1.0/3.0;
	
	h_face(SubSten::north,FaceLabel::north,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::north,FaceLabel::north,nQ,yp1) =  5.0/6.0;
	h_face(SubSten::north,FaceLabel::north,nQ,yp2) = -1.0/6.0;
	
	// Reconstruction weights for west face
	h_face(SubSten::west,FaceLabel::west,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::west,FaceLabel::west,nQ,xm1) =  5.0/6.0;
	h_face(SubSten::west,FaceLabel::west,nQ,xm2) = -1.0/6.0;
	
	h_face(SubSten::center,FaceLabel::west,nQ,xp1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::west,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::west,nQ,xm1) =  1.0/3.0;
	
	h_face(SubSten::east,FaceLabel::west,nQ,xp2) =  1.0/3.0;
	h_face(SubSten::east,FaceLabel::west,nQ,xp1) = -7.0/6.0;
	h_face(SubSten::east,FaceLabel::west,nQ,ctr) = 11.0/6.0;
	
	// Reconstruction weights for south face
	h_face(SubSten::south,FaceLabel::south,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::south,FaceLabel::south,nQ,ym1) =  5.0/6.0;
	h_face(SubSten::south,FaceLabel::south,nQ,ym2) = -1.0/6.0;
	
	h_face(SubSten::center,FaceLabel::south,nQ,yp1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::south,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::south,nQ,ym1) =  1.0/3.0;
	
	h_face(SubSten::north,FaceLabel::south,nQ,yp2) =  1.0/3.0;
	h_face(SubSten::north,FaceLabel::south,nQ,yp1) = -7.0/6.0;
	h_face(SubSten::north,FaceLabel::south,nQ,ctr) = 11.0/6.0;
      }
#else
      for (int nQ=0; nQ<core.SI.nqFace_d; nQ++) {
	const int ctr = 3;
	const int xm1 = 0,xp1 = 6,xm2 = 7,xp2 = 32;
	const int ym1 = 1,yp1 = 5,ym2 = 16,yp2 = 23;
	const int zm1 = 2,zp1 = 4,zm2 = 19,zp2 = 20;
	// Reconstruction weights for east face
	h_face(SubSten::west,FaceLabel::east,nQ,xm2) =  1.0/3.0;
	h_face(SubSten::west,FaceLabel::east,nQ,xm1) = -7.0/6.0;
	h_face(SubSten::west,FaceLabel::east,nQ,ctr) = 11.0/6.0;
	
	h_face(SubSten::center,FaceLabel::east,nQ,xm1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::east,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::east,nQ,xp1) =  1.0/3.0;
	
	h_face(SubSten::east,FaceLabel::east,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::east,FaceLabel::east,nQ,xp1) =  5.0/6.0;
	h_face(SubSten::east,FaceLabel::east,nQ,xp2) = -1.0/6.0;
	
	// Reconstruction weights for north face
	h_face(SubSten::south,FaceLabel::north,nQ,ym2) =  1.0/3.0;
	h_face(SubSten::south,FaceLabel::north,nQ,ym1) = -7.0/6.0;
	h_face(SubSten::south,FaceLabel::north,nQ,ctr) = 11.0/6.0;
	
	h_face(SubSten::center,FaceLabel::north,nQ,ym1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::north,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::north,nQ,yp1) =  1.0/3.0;
	
	h_face(SubSten::north,FaceLabel::north,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::north,FaceLabel::north,nQ,yp1) =  5.0/6.0;
	h_face(SubSten::north,FaceLabel::north,nQ,yp2) = -1.0/6.0;
	
	// Reconstruction weights for top face
	h_face(SubSten::bottom,FaceLabel::top,nQ,zm2) =  1.0/3.0;
	h_face(SubSten::bottom,FaceLabel::top,nQ,zm1) = -7.0/6.0;
	h_face(SubSten::bottom,FaceLabel::top,nQ,ctr) = 11.0/6.0;
	
	h_face(SubSten::center,FaceLabel::top,nQ,zm1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::top,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::top,nQ,zp1) =  1.0/3.0;
	
	h_face(SubSten::top,FaceLabel::top,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::top,FaceLabel::top,nQ,zp1) =  5.0/6.0;
	h_face(SubSten::top,FaceLabel::top,nQ,zp2) = -1.0/6.0;
	
	// Reconstruction weights for west face
	h_face(SubSten::west,FaceLabel::west,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::west,FaceLabel::west,nQ,xm1) =  5.0/6.0;
	h_face(SubSten::west,FaceLabel::west,nQ,xm2) = -1.0/6.0;
	
	h_face(SubSten::center,FaceLabel::west,nQ,xp1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::west,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::west,nQ,xm1) =  1.0/3.0;
	
	h_face(SubSten::east,FaceLabel::west,nQ,xp2) =  1.0/3.0;
	h_face(SubSten::east,FaceLabel::west,nQ,xp1) = -7.0/6.0;
	h_face(SubSten::east,FaceLabel::west,nQ,ctr) = 11.0/6.0;
	
	// Reconstruction weights for south face
	h_face(SubSten::south,FaceLabel::south,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::south,FaceLabel::south,nQ,ym1) =  5.0/6.0;
	h_face(SubSten::south,FaceLabel::south,nQ,ym2) = -1.0/6.0;
	
	h_face(SubSten::center,FaceLabel::south,nQ,yp1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::south,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::south,nQ,ym1) =  1.0/3.0;
	
	h_face(SubSten::north,FaceLabel::south,nQ,yp2) =  1.0/3.0;
	h_face(SubSten::north,FaceLabel::south,nQ,yp1) = -7.0/6.0;
	h_face(SubSten::north,FaceLabel::south,nQ,ctr) = 11.0/6.0;
	
	// Reconstruction weights for bottom face
	h_face(SubSten::bottom,FaceLabel::bottom,nQ,ctr) =  1.0/3.0;
	h_face(SubSten::bottom,FaceLabel::bottom,nQ,zm1) =  5.0/6.0;
	h_face(SubSten::bottom,FaceLabel::bottom,nQ,zm2) = -1.0/6.0;
	
	h_face(SubSten::center,FaceLabel::bottom,nQ,zp1) = -1.0/6.0;
	h_face(SubSten::center,FaceLabel::bottom,nQ,ctr) =  5.0/6.0;
	h_face(SubSten::center,FaceLabel::bottom,nQ,zm1) =  1.0/3.0;
	
	h_face(SubSten::top,FaceLabel::bottom,nQ,zp2) =  1.0/3.0;
	h_face(SubSten::top,FaceLabel::bottom,nQ,zp1) = -7.0/6.0;
	h_face(SubSten::top,FaceLabel::bottom,nQ,ctr) = 11.0/6.0;
      }
#endif      
      // Copy to views on the right memory space
      Kokkos::deep_copy(faceWeights,h_face);
      Kokkos::deep_copy(derivWeights,h_deriv);
    }
    
  } // end namespace Stencil
  
} // end namespace KFVM
