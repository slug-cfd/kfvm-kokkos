#ifndef VECTORVALUEDRA_H_
#define VECTORVALUEDRA_H_

#include <cstdio>
#include <iterator>
#include <vector>
#include <complex>
#include <cmath>

#include <Definitions.H>

#include "../Types.H"
#include "CSEKernel.H"
#include "Monomials.H"
#include "StencilSymmetry.H"

namespace KFVM {

  namespace Stencil {

    class VectorValuedRA
    {
      // Internal matrix class
      // This is not meant to be a super optimal implementation,
      // rather just good enough to support this process
      template<typename T>
      struct Matrix
      {
	const int M,N;
	std::vector<T> mat;

	// Construct matrix of given size filled with zeros
	Matrix(int M_,int N_) : M(M_),N(N_),mat(M*N,0) {}
	// Copy construct matrix
	Matrix(const Matrix& other):
	  M(other.M),
	  N(other.N),
	  mat(other.mat) {}
	// copy contents into existing matrix
	void copy(const Matrix& B) { mat.assign(B.mat.begin(),B.mat.end()); }
	void writeBlock(int dstRowStart,int dstRowEnd,int dstColStart,int dstColEnd,
			const Matrix<T>& src,
			int srcRowStart,int srcRowEnd,int srcColStart,int srcColEnd)
	{
	  int nRows = dstRowEnd - dstRowStart,nCols = dstColEnd - dstColStart;
	  for (int j=0; j<nCols; j++) {
	    int dj = dstColStart + j,sj = srcColStart + j;
	    for (int i=0; i<nRows; i++) {
	      int di = dstRowStart + i,si = srcRowStart + i;
	      mat[dj*M + di] = src(si,sj);
	    }
	  }
	}
	// Access operators
	T& operator()(int i,int j) { return mat[j*M + i]; }
	const T& operator()(int i,int j) const { return mat[j*M + i]; }
	// Return underlying data pointer
	T* data() { return mat.data(); }
      private:
	Matrix() = delete;
      };

      // Enumeration of used monomials for augmented kernel system
      struct MonomialEnum
      {
	const int tDeg,nMono;
	std::vector<std::array<int,SPACE_DIM> > deg;

	MonomialEnum(int tDeg_):
	  tDeg(tDeg_),
	  nMono(SPACE_DIM==2 ? (tDeg + 1)*(tDeg + 2)/2 : (tDeg + 1)*(tDeg + 2)*(tDeg + 3)/6),
	  deg(nMono)
	{
	  // Enumerate all monomials up to total degree tDeg
	  int n = 0;
#if (SPACE_DIM == 2)
	  for (int t=0; t<=tDeg; t++) {
	    for (int i=0; i<=t; i++) {
	      deg[n][0] = i;
	      deg[n][1] = t - i;
	      n++;
	    }
	  }
#else
	  for (int t=0; t<=tDeg; t++) {
	    for (int i=0; i<=t; i++) {
	      for (int j=0; j<=(t - i); j++) {
		deg[n][0] = i;
		deg[n][1] = j;
		deg[n][2] = t - i - j;
		n++;
	      }
	    }
	  }
#endif
	}
      };
      
      // Wrappers for Lapack calls
      static int c_dgemv(char,double,Matrix<double>&,
			 std::vector<double>&,double,std::vector<double>&);
      static int c_dgemm(char,char,double,Matrix<double>&,
			 Matrix<double>&,double,Matrix<double>&);
      static int c_dgeqrf(Matrix<double>&,Matrix<double>&,Matrix<double>&);
      static int c_dtrsm(char,char,char,char,
			 double,Matrix<double>&,std::vector<double>&);
      static int c_dgels(char,Matrix<double>&,std::vector<double>&);
      static int c_zgetrf(Matrix<std::complex<double>>&,std::vector<int>&);
      static int c_zgetrs(char,Matrix<std::complex<double>>&,std::vector<int>&,
			  std::vector<std::complex<double>>&);

      // Stencil
      int stenSize;
      std::vector<double> KFVM_D_DECL(xs,ys,zs);

      // Desired epsilon value
      double eps;

      // Set of monomials to append
      MonomialEnum monos;

      // Size of kernel systems with additional polynomials
      int sysSize;

      // Number of contour points to use
      int K;

      // Numerator and denominator terms
      int N,M;

      // Contour
      std::vector<std::complex<double>> epsCont;

      // System matrices
      std::vector<std::vector<int>> ipiv;
      std::vector<Matrix<std::complex<double>>> C;

      // Helper functions
      void fillCovMat(std::complex<double>,
		      Matrix<std::complex<double>>&);
      
      // Sample vector
      template<KFVM_D_DECL(class efX,class efY,class efZ)>
      void fillSampleVec(std::complex<double> ep,
			 KFVM_D_DECL(const double qx,const double qy,const double qz),
			 std::vector<std::complex<double>>& T)
      {
	// Evaluate kernel at query points for each center
	for (int i=0; i<stenSize; i++) {
	  double KFVM_D_DECL(dx = qx - xs[i],
			     dy = qy - ys[i],
			     dz = qz - zs[i]);
	  T[i] = SE::K<KFVM_D_DECL(efX,efY,efZ)>(ep,KFVM_D_DECL(dx,dy,dz));
	}
	// Evaluate polynomials at query points
	for (int i=0; i<monos.nMono; i++) {
	  T[i + stenSize] =
	    Monomials::mono<KFVM_D_DECL(efX,efY,efZ)>(monos.deg[i],
						      KFVM_D_DECL(qx,qy,qz));
	}
      }

      void fillGs(const std::vector<std::vector<double>>&,
		  std::vector<Matrix<double>>&);

      void fillE(const std::vector<double>&,Matrix<double>&);

      void formFdot(VectorValuedRA::Matrix<double>&,
		    std::vector<std::vector<double>>&,
		    std::vector<VectorValuedRA::Matrix<double>>&,
		    std::vector<std::vector<double>>&,
		    std::vector<VectorValuedRA::Matrix<double>>&,
		    std::vector<double>&,
		    VectorValuedRA::Matrix<double>&);

      double rationalEval(const std::vector<double>&,
			  const std::vector<double>&);

      // Generate prediction vector
      template<KFVM_D_DECL(class efX,class efY,class efZ)>
      void predVec(KFVM_D_DECL(const double qx,const double qy,const double qz),
		   std::vector<double>& pred)
      {
	// Solve system for each epsilon value
	std::vector<std::vector<double>> f(stenSize,std::vector<double>(K,0.0));
	std::vector<std::complex<double> > fc(sysSize,std::complex<double>(0.0,0.0));
	std::vector<double> f_infty(K/2,0.0);
	for (int k=0; k<K/2; k++) {
	  fillSampleVec<KFVM_D_DECL(efX,efY,efZ)>(epsCont[k],
						  KFVM_D_DECL(qx,qy,qz),fc);
	  c_zgetrs('N',C[k],ipiv[k],fc);
	  
	  // get infinity norm of prediction vector
	  // Note that the tail of fc associated to the polynomial terms
	  // is ignored
	  for (int j=0; j<stenSize; j++) {
	    f_infty[k] = std::fmax(f_infty[k],std::abs(fc[j]));
	  }

	  // Scatter into real valued pred vecs and rescale
	  // Polynomial tail is still ignored
	  for (int j=0; j<stenSize; j++) {
	    f[j][k      ] = fc[j].real()/f_infty[k];
	    f[j][k + K/2] = fc[j].imag()/f_infty[k];
	  }
	}
	
	// Fill data matrices
	std::vector<Matrix<double>> G(stenSize,{K,N});
	fillGs(f,G);
      
	// Fill scaled E matrix
	Matrix<double> E(K,M + 1);
	fillE(f_infty,E);

	// QR factorize E matrix
	Matrix<double> Q(K,K),R(M + 1,M + 1);
	c_dgeqrf(E,Q,R);

	// Multiply G's and f by Q^T
	// Rearrange into Fdot and qtf
	std::vector<std::vector<double>> qtf(stenSize,std::vector<double>(M + 1,0.0));
	std::vector<Matrix<double>> Gr(stenSize,{M + 1,N});
	std::vector<double> fdot(stenSize*(K - M - 1),0.0);
	Matrix<double> Fdot(stenSize*(K - M - 1),N);
	formFdot(Q,f,G,qtf,Gr,fdot,Fdot);

	// Least-squares solve for denominator coefficients
	c_dgels('N',Fdot,fdot);

	// Move all Gr[j]*b vectors to RHS,
	// solve for numerator coefficients,
	// and evaluate rational function at the desired epsilon
	for (int j=0; j<stenSize; j++) {
	  c_dgemv('N',-1.0,Gr[j],fdot,1.0,qtf[j]);
	  c_dtrsm('L','U','N','N',1.0,R,qtf[j]);
	  pred[j] = rationalEval(qtf[j],fdot);
	}
      }
      
    public:
      VectorValuedRA(double,KFVM_D_DECL(const std::vector<double>&,
					const std::vector<double>&,
					const std::vector<double>&));
      
      template<class PredView,KFVM_D_DECL(class efX,class efY,class efZ)>
      void predVecs(KFVM_D_DECL(const std::vector<double>& qx,
				const std::vector<double>& qy,
				const std::vector<double>& qz),
		    PredView wts)
      {
	// Generate symmetries for the set of query points
	StencilSymmetry stenSym(true,
				KFVM_D_DECL(xs,ys,zs),KFVM_D_DECL(xs,ys,zs),
				KFVM_D_DECL(qx,qy,qz),KFVM_D_DECL(qx,qy,qz));
	
	// Generate prediction vector for each query point
	std::vector<double> pred(stenSize,0.0);
	for (int nQ=0; nQ<qx.size(); nQ++) {
	  if (stenSym.symMap[nQ].first < 0) {
	    // Create new prediction vector if no symmetry relation is available
	    predVec<KFVM_D_DECL(efX,efY,efZ)>(KFVM_D_DECL(qx[nQ],qy[nQ],qz[nQ]),pred);
	    for (int j=0; j<stenSize; j++) {
	      wts(nQ,j) = static_cast<Real>(pred[j]);
	    }
	  } else {
	    for (int j=0; j<stenSize; j++) {
	      wts(nQ,j) = wts(stenSym.symMap[nQ].first,stenSym.idxMap[nQ][j]);
	    }
	  }
	}
      }
    };
    
  } // end namespace Stencil

} // namespace KFVM

#endif
