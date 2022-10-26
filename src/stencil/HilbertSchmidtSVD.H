#ifndef HILBERTSCHMIDTSVD_H_
#define HILBERTSCHMIDTSVD_H_


#include <cstdio>
#include <array>
#include <vector>
#include <cmath>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"
#include "HermitePolynomials.H"
#include "SEKernel.H"

namespace KFVM {

  namespace Stencil {

    class HS_SVD
    {
      // Internal matrix class
      // This is not meant to be a super optimal implementation,
      // rather just good enough to support this process
      struct Matrix
      {
	const int M,N;
	std::vector<double> mat;

	// Construct matrix of given size filled with 0.0
	Matrix(int M_,int N_):M(M_),N(N_),mat(M*N,0.0) {}
	// Copy construct matrix
	Matrix(const Matrix& other):
	  M(other.M),
	  N(other.N),
	  mat(other.mat) {}
	// copy contents into existing matrix
	void copy(const Matrix& B) { mat.assign(B.mat.begin(),B.mat.end()); }
	// Access operators
	double& operator()(int i,int j) { return mat[j*M + i]; }
	const double& operator()(int i,int j) const { return mat[j*M + i]; }
	// Return underlying data pointer
	double* data() { return mat.data(); }
	// Wrappers for Lapack calls
	static int c_dgemm(char,char,double,Matrix&,Matrix&,double,Matrix&);
	static int c_dgesdd(Matrix&,Matrix&,std::vector<double>&,Matrix&);
	static int c_dgetrf(Matrix&,std::vector<int>&);
	static int c_dgetrs(Matrix&,std::vector<int>&,Matrix&);
	static int c_dgetri(Matrix&,std::vector<int>&);
      private:
	Matrix() = delete;
      };

      // HS internal parameters
      struct Parameters
      {
	const double eps,alpha,beta,delsq;
	const int tDeg,nEig;
	std::vector<std::array<int,SPACE_DIM> > eigOrd;

	Parameters(double eps_,double alpha_,int tDeg_):
	  eps(eps_),
	  alpha(alpha_),
	  beta(std::pow(1.0 + 4.0*eps*eps/(alpha*alpha),0.25)),
	  delsq(alpha*alpha*(beta*beta - 1.0)/2.0),
	  tDeg(tDeg_),
	  nEig(SPACE_DIM==2 ? (tDeg + 1)*(tDeg + 2)/2 : (tDeg + 1)*(tDeg + 2)*(tDeg + 3)/6),
	  eigOrd(nEig)
	{
	  // Enumerate all eigenvalues up to total degree tDeg
	  int n = 0;
#if (SPACE_DIM == 2)
	  for (int t=0; t<=tDeg; t++) {
	    for (int i=0; i<=t/2; i++) {
	      eigOrd[n][0] = i;
	      eigOrd[n][1] = t - i;
	      n++;
	      if (t != 2*i) {
		eigOrd[n][0] = t - i;
		eigOrd[n][1] = i;
		n++;
	      }
	    }
	  }
#else
	  for (int t=0; t<=tDeg; t++) {
	    for (int i=0; i<=t; i++) {
	      for (int j=0; j<=(t - i); j++) {
		eigOrd[n][0] = i;
		eigOrd[n][1] = j;
		eigOrd[n][2] = t - i - j;
		n++;
	      }
	    }
	  }
#endif
	}
      };

      // HS Parameters
      Parameters pars;

      // Hermite polynomial evaluator
      HermitePolynomials herm;

      // System sizes
      int N,M;
      
      // Phi matrix and its SVD
      Matrix Phi,U,Vt,V1,V2;
      std::vector<double> S;

      // HS eigenvalues
      std::vector<double> Lam1,Lam2;

      // Correction matrix block and concatenation
      Matrix LVViLi,C;

      // Outer system matrix and its pivots
      Matrix A;
      std::vector<int> ipiv;

      // Helper function to evaluate eigenfunction expansions
      void MercerMats(KFVM_D_DECL(const std::vector<double>&,
				  const std::vector<double>&,
				  const std::vector<double>&));
      
    public:    
      HS_SVD(const double,const int,
	     KFVM_D_DECL(const std::vector<double>&,
			 const std::vector<double>&,
			 const std::vector<double>&));
      
      template<class WtsViewType,KFVM_D_DECL(class FTx,class FTy,class FTz)>
      void predVecs(KFVM_D_DECL(const std::vector<double>& xq,
				const std::vector<double>& yq,
				const std::vector<double>& zq),
		    WtsViewType wts)
      {
	const int Nq = xq.size();
	int lerr = 0;
	
	// Create query point phi matrix (already transposed)
	Matrix PhiQ(M,Nq);
	for (int m=0; m<M; m++) {
	  for (int n=0; n<Nq; n++) {
	    FTx ftx;
	    PhiQ(m,n) = SE::phi(ftx,xq[n],herm,pars.eigOrd[m][0],
				pars.alpha,pars.beta,pars.delsq);
            FTy fty;
            PhiQ(m,n) *= SE::phi(fty,yq[n],herm,pars.eigOrd[m][1],
				 pars.alpha,pars.beta,pars.delsq);
#if (SPACE_DIM == 3)
	    FTz ftz;
	    PhiQ(m,n) *= SE::phi(ftz,zq[n],herm,pars.eigOrd[m][2],
				 pars.alpha,pars.beta,pars.delsq);
#endif
	  }
	}

	// Generate Psi matrix (already transposed)
	Matrix PsiHat(N,Nq);
	lerr = Matrix::c_dgemm('T','N',1.0,C,PhiQ,0.0,PsiHat);
	if (lerr !=0 ) {
	  std::printf("Error: HS_SVD C*PhiQ failed with code: %d\n",lerr);
	}
	
	// Solve outer system
	lerr = Matrix::c_dgetrs(A,ipiv,PsiHat);

	// Solve inner system
	Matrix Psi(N,Nq);
	for (int n=0; n<N; n++) {
	  for (int nq=0; nq<Nq; nq++) {
	    PsiHat(n,nq) /= S[n];
	  }
	}
	lerr = Matrix::c_dgemm('N','N',1.0,U,PsiHat,0.0,Psi);
	if (lerr !=0 ) {
	  std::printf("Error: HS_SVD U*PhiHat failed with code: %d\n",lerr);
	}

	// Copy out to wts view and cast as needed
	for (int n=0; n<N; n++) {
	  for (int nq=0; nq<Nq; nq++) {
	    wts(nq,n) = static_cast<Real>(Psi(n,nq));
	  }
	}
      }
    };
    
  } // end namespace Stencil

} // end namespace KFVM

#endif
