#include "SEKernel.H"
#include <HermitePolynomials.H>
#include <HilbertSchmidtSVD.H>
#include <cstdint>

namespace KFVM {

  namespace Stencil {

    extern "C" void dgemm_(char*,char*,          // transa,transb
			   int*,int*,int*,       // m,n,k
			   double*,double*,int*, // alpha,A,lda
			   double*,int*,         // B,ldb
			   double*,double*,int*, // beta,C,ldc
			   int*);                // info
    
    extern "C" void dgesdd_(char*,int*,int*, // jobz,M,N
			    double*,int*,    // A,lda
			    double*,         // S
			    double*,int*,    // U,ldu
			    double*,int*,    // Vt,ldvt
			    double*,int*,    // work,lwork
			    int*,            // iwork
			    int*);           // info
    
    extern "C" void dgetrf_(int*,int*,    // M,N
			    double*,int*, // A,lda
			    int*,         // ipiv
			    int*);        // info

    extern "C" void dgetrs_(char*,        // trans
			    int*,int*,    // N,NRHS
			    double*,int*, // A,lda
			    int*,         // ipiv
			    double*,int*, // B,ldb
			    int*);        // info

    extern "C" void dgetri_(int*,         // N
			    double*,int*, // A,lda
			    int*,         // ipiv
			    double*,int*, // work,lwork
			    int*);        // info
    
    // Lapack wrappers for matrix class
    int HS_SVD::Matrix::c_dgemm(char trA,char trB,
			double alpha,HS_SVD::Matrix& A,HS_SVD::Matrix& B,
			double beta,HS_SVD::Matrix& C)
    {
      int info = 0;
      int M = C.M,N = C.N;
      int k = (trA=='N' || trA=='n') ? A.N : A.M;
      int lda = (trA=='N' || trA=='n') ? M : k;
      int ldb = (trB=='N' || trB=='n') ? k : N;
      dgemm_(&trA,&trB,&M,&N,&k,
	     &alpha,A.data(),&lda,B.data(),&ldb,
	     &beta,C.data(),&M,&info);
      return info;
    }
    
    int HS_SVD::Matrix::c_dgesdd(HS_SVD::Matrix& A,
				 HS_SVD::Matrix& U,
				 std::vector<double>& S,
				 HS_SVD::Matrix& Vt)
    {
      int info = 0;
      char jobz = 'S';
      int M = A.M,N = A.N;
      int mind = M < N ? M : N;
      std::vector<int> iwork(8*mind,0);
      int lwork = -1;
      std::vector<double> work(1,0.0);
      // Query work size
      dgesdd_(&jobz,&M,&N,
	      A.data(),&M,
	      S.data(),
	      U.data(),&M,
	      Vt.data(),&M,
	      work.data(),&lwork,
	      iwork.data(),
	      &info);
      // Allocate work and do the computation
      lwork = static_cast<int>(work[0]);
      work.resize(lwork,0.0);
      dgesdd_(&jobz,&M,&N,
	      A.data(),&M,
	      S.data(),
	      U.data(),&M,
	      Vt.data(),&M,
	      work.data(),&lwork,
	      iwork.data(),
	      &info);
      return info;
    }
    
    int HS_SVD::Matrix::c_dgetrf(HS_SVD::Matrix& A,std::vector<int>& ipiv)
    {
      int info = 0,M = A.M,N = A.N;
      dgetrf_(&M,&N,A.data(),&M,ipiv.data(),&info);
      return info;
    }

    int HS_SVD::Matrix::c_dgetrs(HS_SVD::Matrix& A,
				 std::vector<int>& ipiv,
				 HS_SVD::Matrix& B)
    {
      int info = 0,M = A.M,NRHS = B.N;
      char trans = 'N';
      dgetrs_(&trans,
	      &M,&NRHS,
	      A.data(),&M,
	      ipiv.data(),
	      B.data(),&M,
	      &info);
      return info;
    }

    int HS_SVD::Matrix::c_dgetri(HS_SVD::Matrix& A,
				 std::vector<int>& ipiv)
    {
      int info = 0,N = A.N,lwork = -1;
      std::vector<double> work(1,0.0);
      // query for work size
      dgetri_(&N,
	      A.data(),&N,
	      ipiv.data(),
	      work.data(),&lwork,
	      &info);
      // Allocate workspace and run for real
      lwork = static_cast<int>(work[0]);
      work.resize(lwork,0.0);
      dgetri_(&N,
	      A.data(),&N,
	      ipiv.data(),
	      work.data(),&lwork,
	      &info);
      return info;
    }

    // HS_SVD sets up as much as possible right away    
    HS_SVD::HS_SVD(const double eps,const int R,
		   KFVM_D_DECL(const std::vector<double>& xs,
			       const std::vector<double>& ys,
			       const std::vector<double>& zs)):
      pars(eps,1.0,static_cast<int>(3*(2*R + 1))),
      herm(pars.tDeg),
      N(xs.size()),
      M(pars.nEig),
      Phi(N,M),
      U(N,N),
      Vt(N,M),
      V1(N,N),
      V2(M - N,N),
      S(N,0.0),
      Lam1(N,0.0),
      Lam2(M - N,0.0),
      LVViLi(M - N,N),
      C(M,N),
      A(N,N),
      ipiv(N,0)
    { 
      // Lapack error codes
      int lerr = 0;
      
      // Fill Phi and Lambda
      MercerMats(KFVM_D_DECL(xs,ys,zs));

      // Compute SVD of Phi
      lerr = Matrix::c_dgesdd(Phi,U,S,Vt);

      // Unpack Vt into V1 and V2
      for (int j=0; j<N; j++) {
	for (int i=0; i<N; i++) {
	  V1(i,j) = Vt(j,i);
	}
	for (int i=0; i<(M - N); i++) {
	  V2(i,j) = Vt(j,i + N);
	}
      }

      // Invert V1
      Matrix Vi(N,N);
      Vi.copy(V1);
      lerr = Matrix::c_dgetrf(Vi,ipiv);
      lerr = Matrix::c_dgetri(Vi,ipiv);

      // Fill LVViLi
      lerr = Matrix::c_dgemm('N','N',1.0,V2,Vi,0.0,LVViLi);
      for (int j=0; j<N; j++) {
	for (int i=0; i<(M-N); i++) {
	  LVViLi(i,j) *= (Lam2[i]/Lam1[j]);
	}
      }

      // Fill C
      for (int j=0; j<N; j++) {
	C(j,j) = 1.0;
	for (int i=0; i<(M-N); i++) {
	  C(i + N,j) = LVViLi(i,j);
	}
      }

      // Create outer matrix and factor it
      A.copy(V1);
      lerr = Matrix::c_dgemm('T','N',1.0,LVViLi,V2,1.0,A);
      lerr = Matrix::c_dgetrf(A,ipiv);
    }

    void HS_SVD::MercerMats(KFVM_D_DECL(const std::vector<double>& xs,
					const std::vector<double>& ys,
					const std::vector<double>& zs))
    {
      // Fill eigenvalues
      for (int m=0; m<N; m++) {
	Lam1[m] = SE::eig(pars.eigOrd[m][0],
			  pars.eps,pars.alpha,pars.delsq);
	Lam1[m] *= SE::eig(pars.eigOrd[m][1],
			   pars.eps,pars.alpha,pars.delsq);
#if (SPACE_DIM == 3)
	Lam1[m] *= SE::eig(pars.eigOrd[m][2],
			   pars.eps,pars.alpha,pars.delsq);
#endif
      }
      for (int m=0; m<(M - N); m++) {
	Lam2[m] = SE::eig(pars.eigOrd[m + N][0],
			  pars.eps,pars.alpha,pars.delsq);
	Lam2[m] *= SE::eig(pars.eigOrd[m + N][1],
			   pars.eps,pars.alpha,pars.delsq);
#if (SPACE_DIM == 3)
	Lam2[m] *= SE::eig(pars.eigOrd[m + N][2],
			   pars.eps,pars.alpha,pars.delsq);
#endif
      }

      // Fill Phi matrix
      for (int m=0; m<M; m++) {
	for (int n=0; n<N; n++) {
	  SE::PhiFunctional::Average ft;
	  Phi(n,m) = SE::phi(ft,xs[n],herm,pars.eigOrd[m][0],
			     pars.alpha,pars.beta,pars.delsq);
	  Phi(n,m) *= SE::phi(ft,ys[n],herm,pars.eigOrd[m][1],
			      pars.alpha,pars.beta,pars.delsq);
#if (SPACE_DIM == 3)
	  Phi(n,m) *= SE::phi(ft,zs[n],herm,pars.eigOrd[m][2],
			      pars.alpha,pars.beta,pars.delsq);
#endif
	}
      }
    }
  } // end namespace Stencil

} // end namespace KFVM
