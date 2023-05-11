#include <cstdio>
#include <vector>
#include <complex>
#include <cmath>

#include <Definitions.H>

#include "Monomials.H"
#include "VectorValuedRA.H"

namespace KFVM {

  namespace Stencil {
    
    VectorValuedRA::VectorValuedRA(double eps_,
				   KFVM_D_DECL(const std::vector<double>& xs_,
					       const std::vector<double>& ys_,
					       const std::vector<double>& zs_)):
      stenSize(xs_.size()),
      KFVM_D_DECL(xs(xs_),ys(ys_),zs(zs_)),
      eps(eps_),
      monos(1),
      sysSize(stenSize + monos.nMono),
      K(300),
      N(250),
      M(K - N - 1),
      epsCont(K/2,0),
      ipiv(K/2,std::vector<int>(sysSize,0)),
      C(K/2,{sysSize,sysSize})
    {
      // Layout epsilon contour
      double epsRad = 0.4;
      double dT = M_PI/static_cast<double>(K),dhT = dT/2.0;
      for (int k=0; k<K/2; k++) {
	double theta = k*dT + dhT;
	epsCont[k] = std::complex<double>(epsRad*std::cos(theta),epsRad*std::sin(theta));
	
	// Build and factor systems for each epsilon value
	fillCovMat(epsCont[k],C[k]);
	c_zgetrf(C[k],ipiv[k]);
      }
    }

    void VectorValuedRA::fillGs(const std::vector<std::vector<double>>& f,
				std::vector<Matrix<double>>& G)
    {
      for (int j=0; j<stenSize; j++) {
	for (int k=0; k<K/2; k++) {
	  for (int n=1; n<=N; n++) {
	    std::complex<double> fkj(f[j][k],f[j][k + K/2]);
	    auto entry = -fkj*std::pow(epsCont[k],2*n);
	    // Note f already scaled
	    G[j](k      ,n - 1) = entry.real();
	    G[j](k + K/2,n - 1) = entry.imag();
	  }
	}
      }
    }

    void VectorValuedRA::fillE(const std::vector<double>& f_infty,
			       Matrix<double>& E)
    {
      for (int k=0; k<K/2; k++) {
	E(k      ,0) = 1.0/f_infty[k];
	E(k + K/2,0) = 0.0;
	for (int m=1; m<=M; m++) {
	  auto epsPow = std::pow(epsCont[k],2*m);
	  E(k      ,m) = epsPow.real()/f_infty[k];
	  E(k + K/2,m) = epsPow.imag()/f_infty[k];
	}
      }
    }

    void VectorValuedRA::formFdot(VectorValuedRA::Matrix<double>& Q,
				  std::vector<std::vector<double>>& f,
				  std::vector<VectorValuedRA::Matrix<double>>& G,
				  std::vector<std::vector<double>>& qtf,
				  std::vector<VectorValuedRA::Matrix<double>>& Gr,
				  std::vector<double>& fdot,
				  VectorValuedRA::Matrix<double>& Fdot)
    {
      int zblkRows = K - M - 1;
      for (int j=0; j<stenSize; j++) {
	Matrix<double> qtg(K,N);
	c_dgemm('T','N',1.0,Q,G[j],0.0,qtg);
	// Top of QtG gets saved in Greduced
	Gr[j].writeBlock(0,M + 1,0,N,qtg,0,M + 1,0,N);
	// Bottom of QtG gets saved in Fdot
	Fdot.writeBlock(j*zblkRows,(j + 1)*zblkRows,0,N,qtg,M + 1,K,0,N);

	std::vector<double> qtfl(K,0.0);
	c_dgemv('T',1.0,Q,f[j],0.0,qtfl);
	// Top of Qtfl gets saved in Qtf
	for (int i=0; i<=M; i++) {
	  qtf[j][i] = qtfl[i];
	}
	// Bottom of Qtfl gets slotted into fdot
	for (int i=0; i<zblkRows; i++) {
	  fdot[j*zblkRows + i] = qtfl[i + M + 1];
	}
      }
    }

    double VectorValuedRA::rationalEval(const std::vector<double>& a,
					const std::vector<double>& b)
    {
      int nA = a.size(),nB = b.size();
      double numer = a[nA - 1],denom = b[nB - 1];
      // Note that both only contain even powers
      for (int p=(nA-2); p>=0; p--) {
	numer = numer*eps*eps + a[p];
      }
      for (int p=(nB-2); p>=0; p--) {
	denom = denom*eps*eps + b[p];
      }
      // Denominator normalized to have 1 as constant term
      denom = denom*eps*eps + 1.0;

      return numer/denom;
    }

    
    void VectorValuedRA::fillCovMat(std::complex<double> ep,
				    Matrix<std::complex<double>>& A)
    {
      // Fill kernel matrix block
      for (int i=0; i<stenSize; i++) {
	for (int j=0; j<stenSize; j++) {
	  double KFVM_D_DECL(dx = xs[i] - xs[j],
			     dy = ys[i] - ys[j],
			     dz = zs[i] - zs[j]);
	  A(i,j) = SE::K<KFVM_D_DECL(EvalFunctional::Average,
	                             EvalFunctional::Average,
	                             EvalFunctional::Average)>(ep,
							       KFVM_D_DECL(dx,dy,dz));
	}
      }
      // Fill polynomial blocks
      for (int i=0; i<stenSize; i++) {
	for (int j=0; j<monos.nMono; j++) {
	  A(i,j + stenSize) = Monomials::mono<KFVM_D_DECL(EvalFunctional::Average,EvalFunctional::Average,EvalFunctional::Average)>(monos.deg[j],KFVM_D_DECL(xs[i],ys[i],zs[i]));
	}
      }
      for (int i=0; i<monos.nMono; i++) {
	for (int j=0; j<stenSize; j++) {
	  A(i + stenSize,j) = Monomials::mono<KFVM_D_DECL(EvalFunctional::Average,EvalFunctional::Average,EvalFunctional::Average)>(monos.deg[i],KFVM_D_DECL(xs[j],ys[j],zs[j]));
	}
      }
      // Sanity check that final block is zero
      for (int i=stenSize; i<sysSize; i++) {
	for (int j=stenSize; j<sysSize; j++) {
	  A(i,j) = std::complex<double>(0.0,0.0);
	}
      }
    }
    
    // Wrappers for Lapack/BLAS calls
    extern "C" void dgemv_(char*,                // trans
			   int*,int*,            // m,n
			   double*,double*,int*, // alpha,A,lda
			   double*,int*,         // x,incx
			   double*,double*,int*, // beta,y,ldy
			   int*);                // info
    
    extern "C" void dgemm_(char*,char*,          // transa,transb
			   int*,int*,int*,       // m,n,k
			   double*,double*,int*, // alpha,A,lda
			   double*,int*,         // B,ldb
			   double*,double*,int*, // beta,C,ldc
			   int*);                // info

    extern "C" void dgeqrf_(int*,int*,    // m,n
			    double*,int*, // A,lda
			    double*,      // tau
			    double*,int*, // work,lwork
			    int*);        // info

    extern "C" void dtrsm_(char*,char*,          // side,uplo
			   char*,char*,          // transa,diag
			   int*,int*,            // m,n
			   double*,double*,int*, // alpha,A,lda
			   double*,int*,         // B,ldb
			   int*);                // info

    extern "C" void dgels_(char*,          // trans
			   int*,int*,int*, // m,n,nrhs
			   double*,int*,   // A,lda
			   double*,int*,   // B,ldb
			   double*,int*,   // work,lwork
			   int*);          // info

    extern "C" void zgetrf_(int*,int*,                  // m,n
			    std::complex<double>*,int*, // A,lda
			    int*,                       // ipiv
			    int*);                      // info

    extern "C" void zgetrs_(char*,                      // trans
			    int*,int*,                  // n,nrhs
			    std::complex<double>*,int*, // A,lda
			    int*,                       // ipiv
			    std::complex<double>*,int*, // B,ldb
			    int*);                      // info
    
    
    // Lapack wrappers for matrix class
    int VectorValuedRA::c_dgemv(char tr,
				double alpha,VectorValuedRA::Matrix<double>& A,
				std::vector<double>& x,double beta,
				std::vector<double>& y)
    {
      int info = 0,one = 1;
      int M = A.M,N = A.N;
      
      dgemv_(&tr,&M,&N,&alpha,A.data(),&M,
	     x.data(),&one,&beta,y.data(),&one,&info);
      
      return info;
    }
    
    int VectorValuedRA::c_dgemm(char trA,char trB,
				double alpha,VectorValuedRA::Matrix<double>& A,
				VectorValuedRA::Matrix<double>& B,
				double beta,VectorValuedRA::Matrix<double>& C)
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

    // Note: this returns a *Full sized* square Q matrix,
    // but also a *reduced size* square R matrix
    // Weird combo, but useful here
    int VectorValuedRA::c_dgeqrf(VectorValuedRA::Matrix<double>& A,
				 VectorValuedRA::Matrix<double>& Q,
				 VectorValuedRA::Matrix<double>& R)
    {
      int info = 0;
      int M = A.M,N = A.N;
      int mind = M < N ? M : N;
      std::vector<double> tau(mind,0.0);
      
      // Start with workspace query
      int lwork = -1;
      std::vector<double> work(1,0.0);
      dgeqrf_(&M,&N,A.data(),&M,tau.data(),work.data(),&lwork,&info);
      
      // Reallocate workspace and do the full computation
      lwork = static_cast<int>(work[0]);
      work.resize(lwork,0.0);
      dgeqrf_(&M,&N,A.data(),&M,tau.data(),work.data(),&lwork,&info);

      // Unpack A into Q and R
      if (info == 0) {
	// R is just a straight copy
	for (int i=0; i<mind; i++) {
	  for (int j=0; j<i; j++) {
	    R(i,j) = 0.0;
	  }
	  for (int j=i; j<A.N; j++) {
	    R(i,j) = A(i,j);
	  }
	}

	// Q has to be built out of the householder reflectors
	// There is probably a Lapack routine to do this,
	// but we need a full MxM Q, not the reduced version

	// The A matrix is not needed any longer
	// The rest is easier to write if we go ahead and stick tau in it
	for (int j=0; j<mind; j++) {
	  double st = std::sqrt(tau[j]);
	  A(j,j) = st;
	  for (int i=j+1; i<A.M; i++) {
	    A(i,j) *= st;
	  }
	}

	// Fill Q with final reflector
	for (int i=0; i<A.M; i++) {
	  for (int j=0; j<A.M; j++) {
	    Q(i,j) = i == j ? 1.0 : 0.0;
	  }
	}
	for (int i=mind - 1; i<A.M; i++) {
	  for (int j=mind - 1; j<A.M; j++) {
	    Q(i,j) -= A(i,mind - 1)*A(j,mind - 1);
	  }
	}

	// Now bubble upwards and apply each reflector in turn
	work.resize(A.M,0.0);
	for (int k=mind-2; k>=0; k--) {
	  // Put vk^T * Q into work vector
	  for (int j=k; j<mind; j++) {
	    work[j] = 0.0;
	    for (int i=k; i<mind; i++) {
	      work[j] += A(i,k)*Q(i,j);
	    }
	  }
	  // subtract vk * work^T from Q
	  for (int i=k; i<A.M; i++) {
	    for (int j=k; j<A.M; j++) {
	      Q(i,j) -= A(i,k)*work[j];
	    }
	  }
	}
      }
      
      return info;
    }

    int VectorValuedRA::c_dtrsm(char side,char uplo,char transa,char diag,
				double alpha,VectorValuedRA::Matrix<double>& A,
				std::vector<double>& b)
    {
      int info = 0,M = b.size(),one = 1;

      dtrsm_(&side,&uplo,&transa,&diag,
	     &M,&one,&alpha,A.data(),&M,b.data(),&M,&info);

      return info;
    }

    int VectorValuedRA::c_dgels(char trans,VectorValuedRA::Matrix<double>& A,
				std::vector<double>& b)
    {
      int info = 0,one = 1;
      int M = A.M,N = A.N,ldb = b.size();

      // Do workspace query
      int lwork = -1;
      std::vector<double> work(1,0.0);
      dgels_(&trans,&M,&N,&one,
	     A.data(),&M,b.data(),&ldb,work.data(),&lwork,&info);

      // Reallocate work and solve system
      lwork = static_cast<int>(work[0]);
      work.resize(lwork,0.0);
      dgels_(&trans,&M,&N,&one,
	     A.data(),&M,b.data(),&ldb,work.data(),&lwork,&info);

      return info;
    }

    int VectorValuedRA::c_zgetrf(VectorValuedRA::Matrix<std::complex<double>>& A,
				 std::vector<int>& ipiv)
    {
      int info = 0;
      int M = A.M,N = A.N;

      zgetrf_(&M,&N,A.data(),&M,ipiv.data(),&info);

      return info;
    }

    int VectorValuedRA::c_zgetrs(char trans,
				 VectorValuedRA::Matrix<std::complex<double>>& A,
				 std::vector<int>& ipiv,
				 std::vector<std::complex<double>>& b)
    {
      int info = 0,one = 1;
      int M = A.M;

      zgetrs_(&trans,&M,&one,A.data(),&M,ipiv.data(),b.data(),&M,&info);

      return info;
    }
    
  } // end namespace Stencil
  
} // end namespace KFVM
