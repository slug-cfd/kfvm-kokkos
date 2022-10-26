#include <cstdio>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../SimVar.H"
#include "../Types.H"
#include "../ProblemSetup.H"
#include "../Geometry.H"
#include "../numeric/Numeric.H"
#include "Definitions.H"
#include "EvalFunctionals.H"
#include "CSEKernel.H"
#include "Stencil.H"
#include "StencilSymmetry.H"
#include "VectorValuedRA.H"

namespace KFVM {

  namespace Stencil {

    idx_t Stencil::Core::coord2idx(idx_t i,idx_t j)
    {
      idx_t idx = -1;
      for (idx_t n=0; n<Stencil::Core::SI.nCellsFull; n++) {
	if (i == lOff[n] && j == tOff[n]) {
	  idx = n;
	  break;
	}
      }
      return idx;
    }
    
    idx_t Stencil::Core::coord2idx(idx_t i,idx_t j,idx_t k)
    {
      idx_t idx = -1;
      for (idx_t n=0; n<Stencil::Core::SI.nCellsFull; n++) {
	if (i == lOff[n] && j == tOff[n] && k == ttOff[n]) {
	  idx = n;
	  break;
	}
      }
      return idx;
    }
    
    void Stencil::Core::findStencil()
    {
#if (SPACE_DIM == 2)
      double mjRsq = rad*rad;
      idx_t nC = 1;
      // Assure that central substencil is first
      lOff[0] = 0;
      tOff[0] = 0;
      for (idx_t nL=1; nL<=rad; nL++) {
        // On axis always fits
	lOff[nC] = -nL; tOff[nC] = 0;   nC++;
        lOff[nC] = nL;  tOff[nC] = 0;   nC++;
        lOff[nC] = 0;   tOff[nC] = -nL; nC++;
        lOff[nC] = 0;   tOff[nC] = nL;  nC++;
        // Check if diagonal cells fit in radius
	for (idx_t nR=1; nR<=nL; nR++) {
	  double rsq = 2.0*nR*nR;
	  idx_t lyr = static_cast<idx_t>(ceil(sqrt(rsq)));
	  if (rsq <= mjRsq && lyr == nL) {
	    lOff[nC] = -nR; tOff[nC] = -nR; nC++;
	    lOff[nC] = -nR; tOff[nC] = nR;  nC++;
	    lOff[nC] = nR;  tOff[nC] = -nR; nC++;
	    lOff[nC] = nR;  tOff[nC] = nR;  nC++;
	  }
	}
        // Look at remaining cells to fill circle
	for (idx_t i=2; i<=rad; i++) {
	  for (idx_t j=1; j<i; j++) {
	    double cx = i,cy = j,rsq = cx*cx + cy*cy;
	    idx_t lyr = static_cast<idx_t>(ceil(sqrt(rsq)));
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
      idx_t mjRsq = rad*rad;
      idx_t rm = rad - 1,nC = 0;
      // Assure that central substencil is first
      for (idx_t i=-rm; i<=rm; i++) {
	for (idx_t j=-rm; j<=rm; j++) {
	  for (idx_t k=-rm; k<=rm; k++) {
	    if ((i*i + j*j + k*k) <= rm*rm) {
	      lOff[nC] = i; tOff[nC] = j; ttOff[nC] = k; nC++;
	    }
	  }
	}
      }

      // then add all other cells
      for (idx_t i=-rad; i<=rad; i++) {
	for (idx_t j=-rad; j<=rad; j++) {
	  for (idx_t k=-rad; k<=rad; k++) {
	    idx_t rsq = i*i + j*j + k*k;
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
      idx_t nN = 0,nS = 0,nE = 0,nW = 0;
      for (idx_t h=0; h<=rad; h++) {
        // Set on-axis indices first
        subIdx[0][nW++] = coord2idx(-h, 0);
        subIdx[1][nE++] = coord2idx( h, 0);
        subIdx[2][nS++] = coord2idx( 0,-h);
        subIdx[3][nN++] = coord2idx( 0, h);
      }
      // Set off axis indices in pairs 
      for (idx_t h=0; h<=rad; h++) {
        for (idx_t l=1; l<=h; l++) {
	  idx_t idx  = coord2idx(-h,-l);
	  idx_t idxp = coord2idx(-h, l);
          if (idx >= 0) {
            subIdx[0][nW++] = idx;
            subIdx[0][nW++] = idxp;
	    
	    idx_t idx  = coord2idx(h,-l);
	    idx_t idxp = coord2idx(h, l);
            subIdx[1][nE++] = idx;
            subIdx[1][nE++] = idxp;
        
            idx  = coord2idx(-l,-h);
            idxp = coord2idx( l,-h);
            subIdx[2][nS++] = idx;
            subIdx[2][nS++] = idxp;
	    
	    idx  = coord2idx(-l,h);
	    idxp = coord2idx( l,h);
            subIdx[3][nN++] = idx;
            subIdx[3][nN++] = idxp;
          }
        }
      }
#else
      idx_t nN = 0,nS = 0,nE = 0,nW = 0,nT = 0,nB = 0;
      for (idx_t n=0; n<SI.nCellsFull; n++) {
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

    void Stencil::off2Double(idx_t nS,idx_t subsize,
			     KFVM_D_DECL(std::vector<double>& xs,
					 std::vector<double>& ys,
					 std::vector<double>& zs))
    {
      xs.resize(subsize,0.0);
      ys.resize(subsize,0.0);
#if (SPACE_DIM == 3)
      zs.resize(subsize,0.0);
#endif
      for (idx_t n=0; n<subsize; n++) {
	idx_t idx = nS<2 ? n : core.subIdx[nS - 2][n];
	xs[n] = static_cast<double>(core.lOff[idx]);
	ys[n] = static_cast<double>(core.tOff[idx]);
#if (SPACE_DIM == 3)
	zs[n] = static_cast<double>(core.ttOff[idx]);
#endif
      }
    }

    namespace {
      template<class WType>
      void testPV(const WType& wts,idx_t numQuad,KFVM_D_DECL(const std::vector<double>& xs,
							     const std::vector<double>& ys,
							     const std::vector<double>& zs))
      {
	for (idx_t nQ=0; nQ<numQuad; nQ++) {
	  double cv = 0.0,KFVM_D_DECL(xv = 0.0,yv = 0.0,zv = 0.0);
	  for (idx_t j=0; j<xs.size(); j++) {
	    cv += wts(nQ,j);
	    xv += xs[j]*wts(nQ,j);
	    yv += ys[j]*wts(nQ,j);
#if (SPACE_DIM == 3)
	    zv += zs[j]*wts(nQ,j);
#endif
	  }
#if (SPACE_DIM == 2)
          std::printf("    nQ %d: (c,x,y) = (%lf,%lf,%lf)\n",nQ,cv,xv,yv);
#else
          std::printf("    nQ %d: (c,x,y,z) = (%lf,%lf,%lf,%lf)\n",nQ,cv,xv,yv,zv);
#endif
	}
      }
      
      template<class WType>
      void normalizePV_0(const WType& wts,idx_t numQuad,idx_t stenSize)
      {
	for (idx_t nQ=0; nQ<numQuad; nQ++) {
	  Real cv = 0.0;
	  for (idx_t j=0; j<stenSize; j++) {
	    cv += wts(nQ,j);
	  }
          cv /= stenSize;
	  for (idx_t j=0; j<stenSize; j++) {
	    wts(nQ,j) = wts(nQ,j) - cv;
	  }
	}
      }
      
      template<class WType>
      void normalizePV_1(const WType& wts,idx_t numQuad,idx_t stenSize)
      {
	for (idx_t nQ=0; nQ<numQuad; nQ++) {
	  Real cv = 0.0;
	  for (idx_t j=0; j<stenSize; j++) {
	    cv += wts(nQ,j);
	  }
	  for (idx_t j=0; j<stenSize; j++) {
	    wts(nQ,j) = wts(nQ,j)/cv;
	  }
	}
      }
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
      for (idx_t nQ=0; nQ<core.SI.nqFace; nQ++) {
	for (idx_t nR=0; nR<core.SI.nqFace; nR++) {
	  idx_t idx = nQ*core.SI.nqFace + nR;
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
      for (idx_t nQ=0; nQ<core.SI.nqCell; nQ++) {
	for (idx_t nR=0; nR<core.SI.nqCell; nR++) {
	  idx_t idx = nQ*core.SI.nqCell + nR;
	  cq1[idx] = qrc.ab[nQ];
	  cq2[idx] = qrc.ab[nR];
	}
      }
#else
      std::vector<double> cq3(core.SI.nqCell_d,0.0);
      for (idx_t nQ=0; nQ<core.SI.nqCell; nQ++) {
	for (idx_t nR=0; nR<core.SI.nqCell; nR++) {
	  for (idx_t nS=0; nS<core.SI.nqCell; nS++) {
	    idx_t idx = nQ*core.SI.nqCell*core.SI.nqCell + nR*core.SI.nqCell + nS;
	    cq1[idx] = qrc.ab[nQ];
	    cq2[idx] = qrc.ab[nR];
	    cq3[idx] = qrc.ab[nS];
	  }
	}
      }
#endif

      // find weights for centered substencils
      for (idx_t nS=0; nS<2; nS++) {
	// stencil as double arrays
	idx_t subsize = nS==0 ? core.SI.nCellsFull : core.SI.nCellsCtr;
	std::vector<double> KFVM_D_DECL(xs,ys,zs);
	off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));
	
	// Vector valued rational approximation for this (sub)stencil
	VectorValuedRA vvra(eps,KFVM_D_DECL(xs,ys,zs));
	
	// Find weights on west face
	auto wWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	vvra.predVecs<decltype(wWts),KFVM_D_DECL(EvalFunctional::Point,
	                                         EvalFunctional::Point,
	                                         EvalFunctional::Point)>(KFVM_D_DECL(mhalf,fq1,fq2),wWts);
        normalizePV_1(wWts,core.SI.nqFace_d,subsize);
	// testPV(wWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	// Map weights from west face to other faces
	auto eWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry ewStenSym(false,
				  KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				  KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(mhalf,fq1,fq2));
	ewStenSym.mapWeights(eWts,wWts);
	// testPV(eWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto sWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry swStenSym(false,
				  KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				  KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(mhalf,fq1,fq2));
	swStenSym.mapWeights(sWts,wWts);
	// testPV(sWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto nWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry nsStenSym(false,
				  KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				  KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	nsStenSym.mapWeights(nWts,sWts);
	// testPV(nWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	auto bWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry bwStenSym(false,
				  KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				  KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(mhalf,fq1,fq2));
	bwStenSym.mapWeights(bWts,wWts);
	// testPV(bWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto tWts = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry tbStenSym(false,
				  KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				  KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(fq1,fq2,mhalf));
	tbStenSym.mapWeights(tWts,bWts);
	// testPV(tWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif

	// Find weights for each derivative type
	idx_t nQCD = core.SI.nqCell_d;
	auto dxWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dxWts),KFVM_D_DECL(EvalFunctional::Deriv,
						  EvalFunctional::Point,
						  EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxWts);
        normalizePV_0(dxWts,core.SI.nqFace_d,subsize);
	// testPV(dxWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto dxxWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dxxWts),KFVM_D_DECL(EvalFunctional::SecDeriv,
						   EvalFunctional::Point,
						   EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxxWts);
        normalizePV_0(dxxWts,core.SI.nqFace_d,subsize);
	// testPV(dxxWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto dyWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dyWts),KFVM_D_DECL(EvalFunctional::Point,
	                                          EvalFunctional::Deriv,
	                                          EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyWts);
        normalizePV_0(dyWts,core.SI.nqFace_d,subsize);
	// testPV(dyWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto dyyWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dyyWts),KFVM_D_DECL(EvalFunctional::Point,
						   EvalFunctional::SecDeriv,
						   EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyyWts);
        normalizePV_0(dyyWts,core.SI.nqFace_d,subsize);
	// testPV(dyyWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	auto dzWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dzWts),KFVM_D_DECL(EvalFunctional::Point,
						  EvalFunctional::Point,
						  EvalFunctional::Deriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzWts);
        normalizePV_0(dzWts,core.SI.nqFace_d,subsize);
	// testPV(dzWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
        
	auto dzzWts = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dzzWts),KFVM_D_DECL(EvalFunctional::Point,
	                                           EvalFunctional::Point,
	                                           EvalFunctional::SecDeriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzzWts);
        normalizePV_0(dzzWts,core.SI.nqFace_d,subsize);
	// testPV(dzzWts,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
      }

      // Find weights for biased substencils
      {
	// First find west stencil
	idx_t nS = 2;
	
	// stencil as double arrays
	idx_t subsize = core.SI.nCellsBias;
	std::vector<double> KFVM_D_DECL(xw,yw,zw);
	off2Double(nS,subsize,KFVM_D_DECL(xw,yw,zw));
	
	// Vector valued rational approximation for this substencil
	VectorValuedRA vvra(eps,KFVM_D_DECL(xw,yw,zw));
	
	// Find weights on each face
	auto wFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	vvra.predVecs<decltype(wFace_wSten),KFVM_D_DECL(EvalFunctional::Point,
							EvalFunctional::Point,
							EvalFunctional::Point)>(KFVM_D_DECL(mhalf,fq1,fq2),wFace_wSten);
        normalizePV_1(wFace_wSten,core.SI.nqFace_d,subsize);
	// testPV(wFace_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));
	
	auto eFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	vvra.predVecs<decltype(eFace_wSten),KFVM_D_DECL(EvalFunctional::Point,
							EvalFunctional::Point,
							EvalFunctional::Point)>(KFVM_D_DECL(half,fq1,fq2),eFace_wSten);
        normalizePV_1(eFace_wSten,core.SI.nqFace_d,subsize);
	// testPV(eFace_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto sFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	vvra.predVecs<decltype(sFace_wSten),KFVM_D_DECL(EvalFunctional::Point,
							EvalFunctional::Point,
							EvalFunctional::Point)>(KFVM_D_DECL(fq1,mhalf,fq2),sFace_wSten);
        normalizePV_1(sFace_wSten,core.SI.nqFace_d,subsize);
	// testPV(sFace_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto nFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry nsSym(false,
			      KFVM_D_DECL(xw,yw,zw),KFVM_D_DECL(xw,yw,zw),
			      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	nsSym.mapWeights(nFace_wSten,sFace_wSten);
	// testPV(nFace_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

#if (SPACE_DIM == 3)
	auto bFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry bsSym(false,
			      KFVM_D_DECL(xw,yw,zw),KFVM_D_DECL(xw,yw,zw),
			      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(fq1,mhalf,fq2));
	bsSym.mapWeights(bFace_wSten,sFace_wSten);
	// testPV(bFace_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto tFace_wSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	StencilSymmetry tbSym(false,
			      KFVM_D_DECL(xw,yw,zw),KFVM_D_DECL(xw,yw,zw),
			      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(fq1,fq2,mhalf));
	tbSym.mapWeights(tFace_wSten,bFace_wSten);
#endif

	// Find weights for each derivative type
	idx_t nQCD = core.SI.nqCell_d;
	auto dxWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dxWts_wSten),KFVM_D_DECL(EvalFunctional::Deriv,
							EvalFunctional::Point,
							EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxWts_wSten);
	// testPV(dxWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto dxxWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dxxWts_wSten),KFVM_D_DECL(EvalFunctional::SecDeriv,
	                                                 EvalFunctional::Point,
	                                                 EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dxxWts_wSten);
	// testPV(dxxWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));
	
	auto dyWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dyWts_wSten),KFVM_D_DECL(EvalFunctional::Point,
							EvalFunctional::Deriv,
	                                                EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyWts_wSten);
	// testPV(dyWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto dyyWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dyyWts_wSten),KFVM_D_DECL(EvalFunctional::Point,
							 EvalFunctional::SecDeriv,
							 EvalFunctional::Point)>(KFVM_D_DECL(cq1,cq2,cq3),dyyWts_wSten);
	// testPV(dyyWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

#if (SPACE_DIM == 3)
	auto dzWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dzWts_wSten),KFVM_D_DECL(EvalFunctional::Point,
							EvalFunctional::Point,
							EvalFunctional::Deriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzWts_wSten);
	// testPV(dzWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));

	auto dzzWts_wSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	vvra.predVecs<decltype(dzzWts_wSten),KFVM_D_DECL(EvalFunctional::Point,
							 EvalFunctional::Point,
	                                                 EvalFunctional::SecDeriv)>(KFVM_D_DECL(cq1,cq2,cq3),dzzWts_wSten);
	// testPV(dzzWts_wSten,core.SI.nqFace_d,KFVM_D_DECL(xw,yw,zw));
#endif

	// Use west substencil to fill east substencil
	{
	  nS = 3;
          
	  // stencil as double arrays
	  std::vector<double> KFVM_D_DECL(xs,ys,zs);
	  off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));
	
	  auto wFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Wface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(mhalf,fq1,fq2),KFVM_D_DECL(half,fq1,fq2));
	  ewSym_Wface.mapWeights(wFace_eSten,eFace_wSten);
	  // testPV(wFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto eFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Eface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(mhalf,fq1,fq2));
	  ewSym_Eface.mapWeights(eFace_eSten,wFace_wSten);
	  // testPV(eFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	  
	  auto sFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Sface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	  ewSym_Sface.mapWeights(sFace_eSten,sFace_wSten);
	  // testPV(sFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto nFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Nface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(fq1,half,fq2));
	  ewSym_Nface.mapWeights(nFace_eSten,nFace_wSten);
	  // testPV(nFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto bFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Bface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(fq1,fq2,mhalf));
	  ewSym_Bface.mapWeights(bFace_eSten,bFace_wSten);
	  // testPV(bFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto tFace_eSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry ewSym_Tface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(fq1,fq2,half));
	  ewSym_Tface.mapWeights(tFace_eSten,tFace_wSten);
	  // testPV(tFace_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	  auto dxWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	  StencilSymmetry ewSym_deriv(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(cq1,cq2,cq3),KFVM_D_DECL(cq1,cq2,cq3));
	  ewSym_deriv.mapWeights(dxWts_eSten,dxWts_wSten);
	  // testPV(dxWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dxxWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	  ewSym_deriv.mapWeights(dxxWts_eSten,dxxWts_wSten);
	  // testPV(dxxWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	  ewSym_deriv.mapWeights(dyWts_eSten,dyWts_wSten);
	  // testPV(dyWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyyWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	  ewSym_deriv.mapWeights(dyyWts_eSten,dyyWts_wSten);
	  // testPV(dyyWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto dzWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	  ewSym_deriv.mapWeights(dzWts_eSten,dzWts_wSten);
	  // testPV(dzWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzzWts_eSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	  ewSym_deriv.mapWeights(dzzWts_eSten,dzzWts_wSten);
	  // testPV(dzzWts_eSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	}

	// Use west substencil to fill south substencil
	{
	  nS = 4;
	  
	  // stencil as double arrays
	  std::vector<double> KFVM_D_DECL(xs,ys,zs);
	  off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));
	
	  auto wFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Wface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(mhalf,fq1,fq2),KFVM_D_DECL(fq1,half,fq2));
	  swSym_Wface.mapWeights(wFace_sSten,nFace_wSten);
	  // testPV(wFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto eFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Eface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	  swSym_Eface.mapWeights(eFace_sSten,sFace_wSten);
	  // testPV(eFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto sFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Sface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(mhalf,fq1,fq2));
	  swSym_Sface.mapWeights(sFace_sSten,wFace_wSten);
	  // testPV(sFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto nFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Nface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(half,fq1,fq2));
	  swSym_Nface.mapWeights(nFace_sSten,eFace_wSten);
	  // testPV(nFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto bFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Bface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(fq1,fq2,mhalf));
	  swSym_Bface.mapWeights(bFace_sSten,bFace_wSten);
	  // testPV(bFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto tFace_sSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry swSym_Tface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(fq1,fq2,half));
	  swSym_Tface.mapWeights(tFace_sSten,tFace_wSten);
	  // testPV(tFace_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	
	  auto dxWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	  StencilSymmetry swSym_deriv(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(cq1,cq2,cq3),KFVM_D_DECL(cq1,cq2,cq3));
	  swSym_deriv.mapWeights(dxWts_sSten,dxWts_wSten);
	  // testPV(dxWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dxxWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	  swSym_deriv.mapWeights(dxxWts_sSten,dxxWts_wSten);
	  // testPV(dxxWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	  swSym_deriv.mapWeights(dyWts_sSten,dyWts_wSten);
	  // testPV(dyWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyyWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	  swSym_deriv.mapWeights(dyyWts_sSten,dyyWts_wSten);
	  // testPV(dyyWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto dzWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	  swSym_deriv.mapWeights(dzWts_sSten,dzWts_wSten);
	  // testPV(dzWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzzWts_sSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	  swSym_deriv.mapWeights(dzzWts_sSten,dzzWts_wSten);
	  // testPV(dzzWts_sSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	}

	// Use west substencil to fill north substencil
	{
	  nS = 5;
	  // stencil as double arrays
	  std::vector<double> KFVM_D_DECL(xs,ys,zs);
	  off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));
	
	  auto wFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Wface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(mhalf,fq1,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	  nwSym_Wface.mapWeights(wFace_nSten,sFace_wSten);
	  // testPV(wFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto eFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Eface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(fq1,half,fq2));
	  nwSym_Eface.mapWeights(eFace_nSten,nFace_wSten);
	  // testPV(eFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto sFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Sface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(half,fq1,fq2));
	  nwSym_Sface.mapWeights(sFace_nSten,eFace_wSten);
	  // testPV(sFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto nFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Nface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(mhalf,fq1,fq2));
	  nwSym_Nface.mapWeights(nFace_nSten,wFace_wSten);
	  // testPV(nFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto bFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Bface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(fq1,fq2,mhalf));
	  nwSym_Bface.mapWeights(bFace_nSten,bFace_wSten);
	  // testPV(bFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto tFace_nSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry nwSym_Tface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(fq1,fq2,half));
	  nwSym_Tface.mapWeights(tFace_nSten,tFace_wSten);
	  // testPV(tFace_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	
	  auto dxWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	  StencilSymmetry nwSym_deriv(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(cq1,cq2,cq3),KFVM_D_DECL(cq1,cq2,cq3));
	  nwSym_deriv.mapWeights(dxWts_nSten,dxWts_wSten);
	  // testPV(dxWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dxxWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	  nwSym_deriv.mapWeights(dxxWts_nSten,dxxWts_wSten);
	  // testPV(dxxWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	  nwSym_deriv.mapWeights(dyWts_nSten,dyWts_wSten);
	  // testPV(dyWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyyWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	  nwSym_deriv.mapWeights(dyyWts_nSten,dyyWts_wSten);
	  // testPV(dyyWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

#if (SPACE_DIM == 3)
	  auto dzWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	  nwSym_deriv.mapWeights(dzWts_nSten,dzWts_wSten);
	  // testPV(dzWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzzWts_nSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	  nwSym_deriv.mapWeights(dzzWts_nSten,dzzWts_wSten);
	  // testPV(dzzWts_nSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
#endif
	}

#if (SPACE_DIM == 3)
	// Use west substencil to fill bottom substencil
	{
	  nS = 6;
	  // stencil as double arrays
	  std::vector<double> KFVM_D_DECL(xs,ys,zs);
	  off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));
	
	  auto wFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Wface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(mhalf,fq1,fq2),KFVM_D_DECL(fq1,fq2,half));
	  bwSym_Wface.mapWeights(wFace_bSten,tFace_wSten);
	  // testPV(wFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto eFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Eface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(fq1,fq2,mhalf));
	  bwSym_Eface.mapWeights(eFace_bSten,bFace_wSten);
	  // testPV(eFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto sFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Sface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	  bwSym_Sface.mapWeights(sFace_bSten,sFace_wSten);
	  // testPV(sFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto nFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Nface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(fq1,half,fq2));
	  bwSym_Nface.mapWeights(nFace_bSten,nFace_wSten);
	  // testPV(nFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	  
	  auto bFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Bface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(mhalf,fq1,fq2));
	  bwSym_Bface.mapWeights(bFace_bSten,wFace_wSten);
	  // testPV(bFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto tFace_bSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry bwSym_Tface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(half,fq1,fq2));
	  bwSym_Tface.mapWeights(tFace_bSten,eFace_wSten);
	  // testPV(tFace_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto dxWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	  StencilSymmetry bwSym_deriv(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(cq1,cq2,cq3),KFVM_D_DECL(cq1,cq2,cq3));
	  bwSym_deriv.mapWeights(dxWts_bSten,dxWts_wSten);
	  // testPV(dxWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dxxWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	  bwSym_deriv.mapWeights(dxxWts_bSten,dxxWts_wSten);
	  // testPV(dxxWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	  bwSym_deriv.mapWeights(dyWts_bSten,dyWts_wSten);
	  // testPV(dyWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyyWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	  bwSym_deriv.mapWeights(dyyWts_bSten,dyyWts_wSten);
	  // testPV(dyyWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	  bwSym_deriv.mapWeights(dzWts_bSten,dzWts_wSten);
	  // testPV(dzWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzzWts_bSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	  bwSym_deriv.mapWeights(dzzWts_bSten,dzzWts_wSten);
	  // testPV(dzzWts_bSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	}
	
	// Use west substencil to fill top substencil
	{
	  nS = 7;
	  
	  // stencil as double arrays
	  std::vector<double> KFVM_D_DECL(xs,ys,zs);
	  off2Double(nS,subsize,KFVM_D_DECL(xs,ys,zs));

	  auto wFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::west),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Wface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(mhalf,fq1,fq2),KFVM_D_DECL(fq1,fq2,mhalf));
	  twSym_Wface.mapWeights(wFace_tSten,bFace_wSten);
	  // testPV(wFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto eFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::east),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Eface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(half,fq1,fq2),KFVM_D_DECL(fq1,fq2,half));
	  twSym_Eface.mapWeights(eFace_tSten,tFace_wSten);
	  // testPV(eFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto sFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::south),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Sface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,mhalf,fq2),KFVM_D_DECL(fq1,mhalf,fq2));
	  twSym_Sface.mapWeights(sFace_tSten,sFace_wSten);
	  // testPV(sFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto nFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::north),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Nface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,half,fq2),KFVM_D_DECL(fq1,half,fq2));
	  twSym_Nface.mapWeights(nFace_tSten,nFace_wSten);
	  // testPV(nFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	  
	  auto bFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::bottom),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Bface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,mhalf),KFVM_D_DECL(half,fq1,fq2));
	  twSym_Bface.mapWeights(bFace_tSten,eFace_wSten);
	  // testPV(bFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto tFace_tSten = Kokkos::subview(h_face,nS,idx_t(FaceLabel::top),Kokkos::ALL,Kokkos::ALL);
	  StencilSymmetry twSym_Tface(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(fq1,fq2,half),KFVM_D_DECL(mhalf,fq1,fq2));
	  twSym_Tface.mapWeights(tFace_tSten,wFace_wSten);
	  // testPV(tFace_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto dxWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(0,nQCD),Kokkos::ALL);
	  StencilSymmetry twSym_deriv(false,
				      KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xw,yw,zw),
				      KFVM_D_DECL(cq1,cq2,cq3),KFVM_D_DECL(cq1,cq2,cq3));
	  twSym_deriv.mapWeights(dxWts_tSten,dxWts_wSten);
	  // testPV(dxWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dxxWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(nQCD,2*nQCD),Kokkos::ALL);
	  twSym_deriv.mapWeights(dxxWts_tSten,dxxWts_wSten);
	  // testPV(dxxWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(2*nQCD,3*nQCD),Kokkos::ALL);
	  twSym_deriv.mapWeights(dyWts_tSten,dyWts_wSten);
	  // testPV(dyWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dyyWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(3*nQCD,4*nQCD),Kokkos::ALL);
	  twSym_deriv.mapWeights(dyyWts_tSten,dyyWts_wSten);
	  // testPV(dyyWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));

	  auto dzWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(4*nQCD,5*nQCD),Kokkos::ALL);
	  twSym_deriv.mapWeights(dzWts_tSten,dzWts_wSten);
	  // testPV(dzWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	
	  auto dzzWts_tSten = Kokkos::subview(h_deriv,nS,std::pair<idx_t,idx_t>(5*nQCD,6*nQCD),Kokkos::ALL);
	  twSym_deriv.mapWeights(dzzWts_tSten,dzzWts_wSten);
	  // testPV(dzzWts_tSten,core.SI.nqFace_d,KFVM_D_DECL(xs,ys,zs));
	}
#endif
      }
      
      // Copy to views on the right memory space
      Kokkos::deep_copy(faceWeights,h_face);
      Kokkos::deep_copy(derivWeights,h_deriv);
    }
    
  } // end namespace Stencil
  
} // end namespace KFVM
