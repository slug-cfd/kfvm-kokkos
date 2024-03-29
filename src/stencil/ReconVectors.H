#ifndef RECONVECTORS_H_
#define RECONVECTORS_H_

#include <cmath>
#include <cstdio>
#include <iterator>
#include <vector>

#include <Definitions.H>

#include "../Types.H"
#include "Kernels.H"
#include "Monomials.H"
#include "StencilSymmetry.H"

namespace KFVM {

namespace Stencil {

class ReconVectors {
  // Internal matrix class
  // This is not meant to be a super optimal implementation
  // It is only used a few times during set up for small matrices
  template <typename T>
  struct Matrix {
    const int M, N;
    std::vector<T> mat;

    // Construct matrix of given size filled with zeros
    Matrix(int M_, int N_) : M(M_), N(N_), mat(M * N, 0) {}
    // Copy construct matrix
    Matrix(const Matrix &other) : M(other.M), N(other.N), mat(other.mat) {}
    // copy contents into existing matrix
    void copy(const Matrix &B) { mat.assign(B.mat.begin(), B.mat.end()); }
    void writeBlock(int dstRowStart, int dstRowEnd, int dstColStart, int dstColEnd,
                    const Matrix<T> &src, int srcRowStart, int srcRowEnd, int srcColStart,
                    int srcColEnd) {
      int nRows = dstRowEnd - dstRowStart, nCols = dstColEnd - dstColStart;
      for (int j = 0; j < nCols; j++) {
        int dj = dstColStart + j, sj = srcColStart + j;
        for (int i = 0; i < nRows; i++) {
          int di = dstRowStart + i, si = srcRowStart + i;
          mat[dj * M + di] = src(si, sj);
        }
      }
    }
    // Access operators
    T &operator()(int i, int j) { return mat[j * M + i]; }
    const T &operator()(int i, int j) const { return mat[j * M + i]; }
    // Return underlying data pointer
    T *data() { return mat.data(); }

   private:
    Matrix() = delete;
  };

  // Wrappers for Lapack calls
  static int c_dgetrf(Matrix<double> &, std::vector<int> &);
  static int c_dgetrs(Matrix<double> &, std::vector<int> &, std::vector<double> &);

  // Stencil
  int stenSize;
  std::vector<double> KFVM_D_DECL(xs, ys, zs);

  // Desired epsilon value
  double eps;

  // Set of monomials to append
  Monomials::MonomialEnum monos;

  // Size of kernel systems with additional polynomials
  int sysSize;

  // System matrices
  std::vector<int> ipiv;
  Matrix<double> C;

  // Helper functions
  void fillRecMat();

  // Sample vector
  template <KFVM_D_DECL(class efX, class efY, class efZ)>
  void fillSampleVec(const std::array<idx_t, SPACE_DIM> &Nd,
                     KFVM_D_DECL(const double qx, const double qy, const double qz),
                     std::vector<double> &T) {
    // Evaluate kernel at query points for each center
    for (int i = 0; i < stenSize; i++) {
      double KFVM_D_DECL(dx = qx - xs[i], dy = qy - ys[i], dz = qz - zs[i]);
      T[i] = Kernel<kernelType>::K<KFVM_D_DECL(efX, efY, efZ)>(eps, Nd,
                                                               KFVM_D_DECL(dx, dy, dz));
    }
    // Evaluate mononomials at query points
    for (int i = 0; i < monos.nMono; i++) {
      T[i + stenSize] = Monomials::mono<KFVM_D_DECL(efX, efY, efZ)>(
          monos.deg[i], Nd, KFVM_D_DECL(qx, qy, qz));
    }
  }

  // Generate reconstruction vector
  template <KFVM_D_DECL(class efX, class efY, class efZ)>
  void recVec(const std::array<idx_t, SPACE_DIM> &Nd,
              KFVM_D_DECL(const double qx, const double qy, const double qz),
              std::vector<double> &rec) {
    // Solve system for single reconstruction vector
    std::vector<double> S(sysSize, 0.0);
    fillSampleVec<KFVM_D_DECL(efX, efY, efZ)>(Nd, KFVM_D_DECL(qx, qy, qz), S);
    c_dgetrs(C, ipiv, S);

    // Copy out reconstruction vector
    for (int j = 0; j < stenSize; j++) {
      rec[j] = S[j];
    }
  }

 public:
  ReconVectors(const double, const idx_t,
               KFVM_D_DECL(const std::vector<double> &, const std::vector<double> &,
                           const std::vector<double> &));

  template <class RecView, KFVM_D_DECL(class efX, class efY, class efZ)>
  void recVecs(const std::array<idx_t, SPACE_DIM> &Nd,
               KFVM_D_DECL(const std::vector<double> &qx, const std::vector<double> &qy,
                           const std::vector<double> &qz),
               RecView wts) {
    // Generate symmetries for the set of query points
    StencilSymmetry stenSym(true, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                            KFVM_D_DECL(qx, qy, qz), KFVM_D_DECL(qx, qy, qz));

    // Generate reconstruction vector for each query point
    std::vector<double> rec(stenSize, 0.0);
    for (int nQ = 0; nQ < qx.size(); nQ++) {
      if (stenSym.symMap[nQ].first < 0) {
        // Create new reconstruction vector if no symmetry relation is available
        recVec<KFVM_D_DECL(efX, efY, efZ)>(Nd, KFVM_D_DECL(qx[nQ], qy[nQ], qz[nQ]), rec);
        for (int j = 0; j < stenSize; j++) {
          wts(nQ, j) = static_cast<Real>(rec[j]);
        }
      } else {
        for (int j = 0; j < stenSize; j++) {
          wts(nQ, j) = wts(stenSym.symMap[nQ].first, stenSym.idxMap[nQ][j]);
        }
      }
    }
  }

  template <class RecView, KFVM_D_DECL(class efX, class efY, class efZ)>
  void recVecs(const std::array<idx_t, SPACE_DIM> &Nd,
               KFVM_D_DECL(const double &qx, const double &qy, const double &qz),
               RecView wts) {
    // Generate reconstruction vector for query point
    std::vector<double> rec(stenSize, 0.0);
    recVec<KFVM_D_DECL(efX, efY, efZ)>(Nd, KFVM_D_DECL(qx, qy, qz), rec);

    // Cast and copy into weight view
    for (int j = 0; j < stenSize; j++) {
      wts(j) = static_cast<Real>(rec[j]);
    }
  }
};

} // end namespace Stencil

} // namespace KFVM

#endif
