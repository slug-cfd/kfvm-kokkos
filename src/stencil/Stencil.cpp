#include "SimVar.H"
#include "Types.H"
#include "../numeric/Numeric.H"
#include "Stencil.H"
#include <Kokkos_View.hpp>
#include <cstdio>
#include <utility>

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
        subIdx[0][nE++] = coord2idx( h, 0);
        subIdx[1][nW++] = coord2idx(-h, 0);
        subIdx[2][nN++] = coord2idx( 0, h);
        subIdx[3][nS++] = coord2idx( 0,-h);
      }
      // Set off axis indices in pairs 
      for (int h=0; h<=rad; h++) {
        for (int l=1; l<=h; l++) {
	  int idx  = coord2idx(h,-l);
	  int idxp = coord2idx(h, l);
          if (idx >= 0) {
            subIdx[0][nE++] = idx;
            subIdx[0][nE++] = idxp;
        
            idx  = coord2idx(-h,-l);
            idxp = coord2idx(-h, l);
            subIdx[1][nW++] = idx;
            subIdx[1][nW++] = idxp;
	    
	    idx  = coord2idx(-l,h);
	    idxp = coord2idx( l,h);
            subIdx[2][nN++] = idx;
            subIdx[2][nN++] = idxp;
        
            idx  = coord2idx(-l,-h);
            idxp = coord2idx( l,-h);
            subIdx[3][nS++] = idx;
            subIdx[3][nS++] = idxp;
          }
        }
      }
#else
      int nN = 0,nS = 0,nE = 0,nW = 0,nT = 0,nB = 0;
      for (int n=0; n<SI.nCellsFull; n++) {
	// check west substencil
	if (-lOff[n] >= abs(tOff[n]) && -lOff[n] >= abs(ttOff[n])) {
	  subIdx[0][nW++] = n;
	}
	
	// check east substencil
	if (lOff[n] >= abs(tOff[n]) && lOff[n] >= abs(ttOff[n])) {
	  subIdx[1][nE++] = n;
	}
	
	// check south substencil
	if (-tOff[n] >= abs(lOff[n]) && -tOff[n] >= abs(ttOff[n])) {
	  subIdx[2][nS++] = n;
	}
	
	// check north substencil
	if (tOff[n] >= abs(lOff[n]) && tOff[n] >= abs(ttOff[n])) {
	  subIdx[3][nN++] = n;
	}
	
	// check bottom substencil
	if (-ttOff[n] >= abs(lOff[n]) && -ttOff[n] >= abs(tOff[n])) {
	  subIdx[4][nB++] = n;
	}
	
	// check top substencil
	if (ttOff[n] >= abs(lOff[n]) && ttOff[n] >= abs(tOff[n])) {
	  subIdx[5][nT++] = n;
	}
      }
#endif
    }

    void Stencil::findWeights(double lfac)
    {
      // Make host accessible views
      auto h_face = Kokkos::create_mirror(faceWeights);
      auto h_deriv = Kokkos::create_mirror(derivWeights);

      // convert length scale to shape parameter
      double eps = 1.0/(lfac*sqrt(2.0));

      // Fill views with zeros initially
      Kokkos::deep_copy(h_face,Real(0.0));
      Kokkos::deep_copy(h_deriv,Real(0.0));
      
      // Flatten quadrature points on faces
      Numeric::QuadRuleLUT<Core::SI.nqFace> qrf;
      std::vector<double> half(core.SI.nqFace_d,0.5);
      std::vector<double> mhalf(core.SI.nqFace_d,-0.5);
      std::vector<double> fq1(core.SI.nqFace_d,0.0);
#if (SPACE_DIM == 2)
      fq1.assign(qrf.ab.begin(),qrf.ab.end());
#else
      std::vector<double> fq2(core.SI.nqFace_d,0.0);
      for (int nQ=0; nQ<core.SI.nqFace; nQ++) {
	for (int nR=0; nR<core.SI.nqFace; nR++) {
	  int idx = nQ*core.SI.nqFace + nR;
	  fq1[idx] = qrf.ab[nQ];
	  fq2[idx] = qrf.ab[nR];
	}
      }
#endif

      // Flatten quadrature points in cell
      Numeric::QuadRuleLUT<Core::SI.nqCell> qrc;
      std::vector<double> cq1(core.SI.nqCell_d,0.0);
      std::vector<double> cq2(core.SI.nqCell_d,0.0);
#if (SPACE_DIM == 2)
      for (int nQ=0; nQ<core.SI.nqCell; nQ++) {
	for (int nR=0; nR<core.SI.nqCell; nR++) {
	  int idx = nQ*core.SI.nqCell + nR;
	  cq1[idx] = qrc.ab[nQ];
	  cq2[idx] = qrc.ab[nR];
	}
      }
#else
      std::vector<double> cq3(core.SI.nqCell_d,0.0);
      for (int nQ=0; nQ<core.SI.nqCell; nQ++) {
	for (int nR=0; nR<core.SI.nqCell; nR++) {
	  for (int nS=0; nS<core.SI.nqCell; nS++) {
	    int idx = nQ*nR*core.SI.nqCell + nR*core.SI.nqCell + nS;
	    cq1[idx] = qrc.ab[nQ];
	    cq2[idx] = qrc.ab[nR];
	    cq3[idx] = qrc.ab[nS];
	  }
	}
      }
#endif

      // substencil sizes for reference
#if (SPACE_DIM == 2)
      int subsize[] = {core.SI.nCellsFull,core.SI.nCellsCtr,
		       core.SI.nCellsBias,core.SI.nCellsBias,
		       core.SI.nCellsBias,core.SI.nCellsBias};
#else
      int subsize[] = {core.SI.nCellsFull,core.SI.nCellsCtr,
		       core.SI.nCellsBias,core.SI.nCellsBias,
		       core.SI.nCellsBias,core.SI.nCellsBias,
		       core.SI.nCellsBias,core.SI.nCellsBias};
#endif

      // find weights one substencil at a time
      for (int nS=0; nS<core.SI.nSub; nS++) {
	// stencil as double arrays
	std::vector<double> xs(subsize[nS],0.0);
	std::vector<double> ys(subsize[nS],0.0);
#if (SPACE_DIM == 3)
	std::vector<double> zs(subsize[nS],0.0);
#endif
	for (int n=0; n<subsize[nS]; n++) {
	  int idx = nS<2 ? n : core.subIdx[nS - 2][n];
	  xs[n] = static_cast<double>(core.lOff[idx]);
	  ys[n] = static_cast<double>(core.tOff[idx]);
#if (SPACE_DIM == 3)
	  zs[n] = static_cast<double>(core.ttOff[idx]);
#endif
	}
	
	// Hilbert-Schmidt object for this stencil
	HS_SVD hs_svd(eps,rad,KFVM_D_DECL(xs,ys,zs));
	
	// Find weights on each face
	auto wWts = Kokkos::subview(h_face,nS,int(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(wWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
				    SE::PhiFunctional::Point,
				    SE::PhiFunctional::Point)>(KFVM_D_DECL(mhalf,fq1,fq2),wWts);
	
	auto eWts = Kokkos::subview(h_face,nS,int(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(eWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
				    SE::PhiFunctional::Point,
				    SE::PhiFunctional::Point)>(KFVM_D_DECL(half,fq1,fq2),eWts);
	
	auto sWts = Kokkos::subview(h_face,nS,int(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(sWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(fq1,mhalf,fq2),sWts);
	
	auto nWts = Kokkos::subview(h_face,nS,int(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(nWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(fq1,half,fq2),nWts);
#if (SPACE_DIM == 3)
	auto bWts = Kokkos::subview(h_face,nS,int(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(bWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(fq1,fq2,mhalf),bWts);
	
	auto tWts = Kokkos::subview(h_face,nS,int(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	hs_svd.predVecs<decltype(tWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(fq1,fq2,half),tWts);
#endif

	// Find weights for each derivative type
	int nQCD = core.SI.nqCell_d;
	auto dxWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(0,nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dxWts),
			KFVM_D_DECL(SE::PhiFunctional::Deriv,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxWts);
	
	auto dxxWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(nQCD,2*nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dxxWts),
			KFVM_D_DECL(SE::PhiFunctional::SecDeriv,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxxWts);
	
	auto dyWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(2*nQCD,3*nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dyWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Deriv,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyWts);
	
	auto dyyWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(3*nQCD,4*nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dyyWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::SecDeriv,
	                            SE::PhiFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyyWts);
#if (SPACE_DIM == 3)	
	auto dzWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(4*nQCD,5*nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dzWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Point,
	                            SE::PhiFunctional::Deriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzWts);
	
	auto dzzWts = Kokkos::subview(h_deriv,nS,std::pair<int,int>(5*nQCD,6*nQCD),Kokkos::ALL);
	hs_svd.predVecs<decltype(dzzWts),
			KFVM_D_DECL(SE::PhiFunctional::Point,
	  SE::PhiFunctional::Point,
	                            SE::PhiFunctional::SecDeriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzzWts);
#endif
      }
      // Copy to views on the right memory space
      Kokkos::deep_copy(faceWeights,h_face);
      Kokkos::deep_copy(derivWeights,h_deriv);
    }
    
  } // end namespace Stencil
  
} // end namespace KFVM
