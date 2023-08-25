#include <cmath>
#include <cstdio>
#include <vector>

#include <Definitions.H>

#include "Monomials.H"
#include "ReconVectors.H"

namespace KFVM {

namespace Stencil {

ReconVectors::ReconVectors(double eps_, int monoDeg_,
                           KFVM_D_DECL(const std::vector<double> &xs_,
                                       const std::vector<double> &ys_,
                                       const std::vector<double> &zs_))
    : stenSize(xs_.size()), KFVM_D_DECL(xs(xs_), ys(ys_), zs(zs_)), eps(eps_),
      monos(monoDeg_), sysSize(stenSize + monos.nMono), ipiv(sysSize, 0),
      C(sysSize, sysSize) {
  // Fill and factor reconstruction matrix
  fillRecMat();
  c_dgetrf(C, ipiv);
}

void ReconVectors::fillRecMat() {
  // Fill kernel matrix block
  for (int i = 0; i < stenSize; i++) {
    for (int j = 0; j < stenSize; j++) {
      double KFVM_D_DECL(dx = xs[i] - xs[j], dy = ys[i] - ys[j], dz = zs[i] - zs[j]);
      C(i, j) = Kernel<kernelType>::K<KFVM_D_DECL(
          EvalFunctional::Average, EvalFunctional::Average, EvalFunctional::Average)>(
          eps, KFVM_D_DECL(dx, dy, dz));
    }
  }
  // Fill polynomial blocks
  for (int i = 0; i < stenSize; i++) {
    for (int j = 0; j < monos.nMono; j++) {
      C(i, j + stenSize) = Monomials::mono<KFVM_D_DECL(
          EvalFunctional::Average, EvalFunctional::Average, EvalFunctional::Average)>(
          monos.deg[j], KFVM_D_DECL(xs[i], ys[i], zs[i]));
      C(j + stenSize, i) = C(i, j + stenSize);
    }
  }
  // Sanity check that final block is zero
  for (int i = stenSize; i < sysSize; i++) {
    for (int j = stenSize; j < sysSize; j++) {
      C(i, j) = 0.0;
    }
  }
}

// Wrappers for Lapack/BLAS calls
extern "C" void dgetrf_(int *, int *,    // m,n
                        double *, int *, // A,lda
                        int *,           // ipiv
                        int *);          // info

extern "C" void dgetrs_(char *,          // trans
                        int *, int *,    // n,nrhs
                        double *, int *, // A,lda
                        int *,           // ipiv
                        double *, int *, // B,ldb
                        int *);          // info

// Lapack wrappers for matrix class
int ReconVectors::c_dgetrf(ReconVectors::Matrix<double> &A, std::vector<int> &ipiv) {
  int info = 0;
  int M = A.M, N = A.N;

  dgetrf_(&M, &N, A.data(), &M, ipiv.data(), &info);

  return info;
}

int ReconVectors::c_dgetrs(ReconVectors::Matrix<double> &A, std::vector<int> &ipiv,
                           std::vector<double> &b) {
  char trans = 'N';
  int info = 0, one = 1;
  int M = A.M;

  dgetrs_(&trans, &M, &one, A.data(), &M, ipiv.data(), b.data(), &M, &info);

  return info;
}

} // end namespace Stencil

} // end namespace KFVM
