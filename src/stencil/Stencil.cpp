#include <utility>
#include <vector>

#include <fmt/core.h>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Geometry.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "../numeric/Numeric.H"
#include "EvalFunctionals.H"
#include "KernelTypes.H"
#include "ReconVectors.H"
#include "Stencil.H"
#include "StencilInfo.H"
#include "StencilSymmetry.H"

namespace KFVM {

namespace Stencil {

idx_t Stencil::Core::coord2idx(idx_t i, idx_t j) {
  idx_t idx = -1;
  for (idx_t n = 0; n < Stencil::Core::SI.nCellsFull; n++) {
    if (i == lOff[n] && j == tOff[n]) {
      idx = n;
      break;
    }
  }
  return idx;
}

idx_t Stencil::Core::coord2idx(idx_t i, idx_t j, idx_t k) {
  idx_t idx = -1;
  for (idx_t n = 0; n < Stencil::Core::SI.nCellsFull; n++) {
    if (i == lOff[n] && j == tOff[n] && k == ttOff[n]) {
      idx = n;
      break;
    }
  }
  return idx;
}

void Stencil::Core::findStencil() {
#if (SPACE_DIM == 2)
  double mjRsq = rad * rad;
  idx_t nC = 1;
  // Assure that central substencil is first
  lOff[0] = 0;
  tOff[0] = 0;
  for (idx_t nL = 1; nL <= rad; nL++) {
    // On axis always fits
    lOff[nC] = -nL;
    tOff[nC] = 0;
    nC++;
    lOff[nC] = nL;
    tOff[nC] = 0;
    nC++;
    lOff[nC] = 0;
    tOff[nC] = -nL;
    nC++;
    lOff[nC] = 0;
    tOff[nC] = nL;
    nC++;
    // Check if diagonal cells fit in radius
    for (idx_t nR = 1; nR <= nL; nR++) {
      double rsq = 2.0 * nR * nR;
      idx_t lyr = static_cast<idx_t>(ceil(sqrt(rsq)));
      if (rsq <= mjRsq && lyr == nL) {
        lOff[nC] = -nR;
        tOff[nC] = -nR;
        nC++;
        lOff[nC] = -nR;
        tOff[nC] = nR;
        nC++;
        lOff[nC] = nR;
        tOff[nC] = -nR;
        nC++;
        lOff[nC] = nR;
        tOff[nC] = nR;
        nC++;
      }
    }
    // Look at remaining cells to fill circle
    for (idx_t i = 2; i <= rad; i++) {
      for (idx_t j = 1; j < i; j++) {
        double cx = i, cy = j, rsq = cx * cx + cy * cy;
        idx_t lyr = static_cast<idx_t>(ceil(sqrt(rsq)));
        if (rsq <= mjRsq && lyr == nL) {
          lOff[nC] = -i;
          tOff[nC] = j;
          nC++;
          lOff[nC] = -i;
          tOff[nC] = -j;
          nC++;
          lOff[nC] = i;
          tOff[nC] = j;
          nC++;
          lOff[nC] = i;
          tOff[nC] = -j;
          nC++;

          lOff[nC] = -j;
          tOff[nC] = i;
          nC++;
          lOff[nC] = -j;
          tOff[nC] = -i;
          nC++;
          lOff[nC] = j;
          tOff[nC] = i;
          nC++;
          lOff[nC] = j;
          tOff[nC] = -i;
          nC++;
        }
      }
    }
  }
#else
  idx_t mjRsq = rad * rad;
  idx_t rm = rad - 1, nC = 0;
  // Assure that central substencil is first
  for (idx_t i = -rm; i <= rm; i++) {
    for (idx_t j = -rm; j <= rm; j++) {
      for (idx_t k = -rm; k <= rm; k++) {
        if ((i * i + j * j + k * k) <= rm * rm) {
          lOff[nC] = i;
          tOff[nC] = j;
          ttOff[nC] = k;
          nC++;
        }
      }
    }
  }

  // then add all other cells
  for (idx_t i = -rad; i <= rad; i++) {
    for (idx_t j = -rad; j <= rad; j++) {
      for (idx_t k = -rad; k <= rad; k++) {
        idx_t rsq = i * i + j * j + k * k;
        if (rsq > rm * rm && rsq <= mjRsq) {
          lOff[nC] = i;
          tOff[nC] = j;
          ttOff[nC] = k;
          nC++;
        }
      }
    }
  }
#endif
  // Find biased substencils
  findSubStencils();
}

void Stencil::Core::findSubStencils() {
  // Assign substencil indices that map into full stencil
#if (SPACE_DIM == 2)
  idx_t nN = 0, nS = 0, nE = 0, nW = 0;
  for (idx_t h = 0; h <= rad; h++) {
    // Set on-axis indices first
    subIdx[0][nW++] = coord2idx(-h, 0);
    subIdx[1][nE++] = coord2idx(h, 0);
    subIdx[2][nS++] = coord2idx(0, -h);
    subIdx[3][nN++] = coord2idx(0, h);
  }
  // Set off axis indices in pairs
  for (idx_t h = 0; h <= rad; h++) {
    for (idx_t l = 1; l <= h; l++) {
      idx_t idx = coord2idx(-h, -l);
      idx_t idxp = coord2idx(-h, l);
      if (idx >= 0) {
        subIdx[0][nW++] = idx;
        subIdx[0][nW++] = idxp;

        idx_t idx = coord2idx(h, -l);
        idx_t idxp = coord2idx(h, l);
        subIdx[1][nE++] = idx;
        subIdx[1][nE++] = idxp;

        idx = coord2idx(-l, -h);
        idxp = coord2idx(l, -h);
        subIdx[2][nS++] = idx;
        subIdx[2][nS++] = idxp;

        idx = coord2idx(-l, h);
        idxp = coord2idx(l, h);
        subIdx[3][nN++] = idx;
        subIdx[3][nN++] = idxp;
      }
    }
  }
#else
  idx_t nN = 0, nS = 0, nE = 0, nW = 0, nT = 0, nB = 0;
  for (idx_t n = 0; n < SI.nCellsFull; n++) {
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

void Stencil::off2Double(idx_t nS, idx_t subsize,
                         KFVM_D_DECL(std::vector<double> &xs, std::vector<double> &ys,
                                     std::vector<double> &zs)) {
  xs.resize(subsize, 0.0);
  ys.resize(subsize, 0.0);
#if (SPACE_DIM == 3)
  zs.resize(subsize, 0.0);
#endif
  for (idx_t n = 0; n < subsize; n++) {
    idx_t idx = nS < 2 ? n : core.subIdx[nS - 2][n];
    xs[n] = static_cast<double>(core.lOff[idx]);
    ys[n] = static_cast<double>(core.tOff[idx]);
#if (SPACE_DIM == 3)
    zs[n] = static_cast<double>(core.ttOff[idx]);
#endif
  }
}

namespace {
template <class WType>
void testPV(const WType &wts, idx_t numQuad,
            KFVM_D_DECL(const std::vector<double> &xs, const std::vector<double> &ys,
                        const std::vector<double> &zs)) {
  for (idx_t nQ = 0; nQ < numQuad; nQ++) {
    double cv = 0.0;
    double KFVM_D_DECL(xv = 0.0, yv = 0.0, zv = 0.0);
    double KFVM_D_DECL(xxv = 0.0, yyv = 0.0, zzv = 0.0);
    for (idx_t j = 0; j < xs.size(); j++) {
      cv += wts(nQ, j);
      xv += xs[j] * wts(nQ, j);
      xxv += Monomials::mono(EvalFunctional::Average(), 2, 0, xs[j]) * wts(nQ, j);
      yv += ys[j] * wts(nQ, j);
      yyv += Monomials::mono(EvalFunctional::Average(), 2, 0, ys[j]) * wts(nQ, j);
#if (SPACE_DIM == 3)
      zv += zs[j] * wts(nQ, j);
      zzv += Monomials::mono(EvalFunctional::Average(), 2, 0, zs[j]) * wts(nQ, j);
#endif
    }
#if (SPACE_DIM == 2)
    fmt::print("    nQ {}: (c,x,y,xx,yy) = ({:<+23},{:<+23},{:<+23},{:<+23},{:<+23})\n",
               nQ, cv, xv, yv, xxv, yyv);
#else
    fmt::print("    nQ {}: (c,x,y,z,xx,yy,zz) = ({},{},{},{},{},{},{})\n", nQ, cv, xv, yv,
               zv, xxv, yyv, zzv);
#endif
  }
}

} // namespace

void Stencil::findWeights(double lfac) {
  typedef EvalFunctional::NDeriv NDeriv;
  const auto &pointSpec = core.derivSet.deg[0];

  // Make host accessible views
  auto h_face = Kokkos::create_mirror(faceWeights);
  auto h_cell = Kokkos::create_mirror(cellWeights);
  auto h_deriv = Kokkos::create_mirror(derivWeights);
  auto h_cellF = Kokkos::View<Real[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nSub]
                                  [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nqCell_d]
                                  [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull],
                              Kokkos::HostSpace>("Stencil::findWeights(h_cellF)");

  // convert length scale to shape parameter
  double eps = 1.0 / (lfac * sqrt(2.0));

  // Fill views with zeros initially
  Kokkos::deep_copy(h_face, Real(0.0));
  Kokkos::deep_copy(h_cell, Real(0.0));
  Kokkos::deep_copy(h_cellF, Real(0.0));
  Kokkos::deep_copy(h_deriv, Real(0.0));

  // Flatten quadrature points on faces
  Numeric::QuadRuleLUT<Core::SI.nqFace> qrf;
  std::vector<double> half(core.SI.nqFace_d, 0.5);
  std::vector<double> mhalf(core.SI.nqFace_d, -0.5);
  std::vector<double> fq1(core.SI.nqFace_d, 0.0);
#if (SPACE_DIM == 2)
  fq1.assign(qrf.ab.begin(), qrf.ab.end());
#else
  std::vector<double> fq2(core.SI.nqFace_d, 0.0);
  for (idx_t nQ = 0; nQ < core.SI.nqFace; nQ++) {
    for (idx_t nR = 0; nR < core.SI.nqFace; nR++) {
      idx_t idx = nQ * core.SI.nqFace + nR;
      fq1[idx] = qrf.ab[nQ];
      fq2[idx] = qrf.ab[nR];
    }
  }
#endif

  // Flatten quadrature points for source terms
  std::vector<double> cq1(core.SI.nqCell_d, 0.0);
  std::vector<double> cq2(core.SI.nqCell_d, 0.0);
#if (SPACE_DIM == 2)
  for (idx_t nQ = 0; nQ < core.SI.nqFace; nQ++) {
    for (idx_t nR = 0; nR < core.SI.nqFace; nR++) {
      idx_t idx = nQ * core.SI.nqFace + nR;
      cq1[idx] = qrf.ab[nQ];
      cq2[idx] = qrf.ab[nR];
    }
  }
#else
  std::vector<double> cq3(core.SI.nqCell_d, 0.0);
  for (idx_t nQ = 0; nQ < core.SI.nqFace; nQ++) {
    for (idx_t nR = 0; nR < core.SI.nqFace; nR++) {
      for (idx_t nS = 0; nS < core.SI.nqFace; nS++) {
        idx_t idx = nQ * core.SI.nqFace * core.SI.nqFace + nR * core.SI.nqFace + nS;
        cq1[idx] = qrf.ab[nQ];
        cq2[idx] = qrf.ab[nR];
        cq3[idx] = qrf.ab[nS];
      }
    }
  }
#endif

  // Derivative quadrature points for smoothness indicators
  // Pretty simple as it is always just the center point
  std::vector<double> KFVM_D_DECL(dq1(1, 0.0), dq2(1, 0.0), dq3(1, 0.0));

  // find weights for centered substencils
  for (idx_t nS = 0; nS < 2; nS++) {
    // stencil as double arrays
    idx_t subSize = nS == 0 ? core.SI.nCellsFull : core.SI.nCellsCtr;
    std::vector<double> KFVM_D_DECL(xs, ys, zs);
    off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

    // Reconstruction vector constructor for this (sub)stencil
    const double kFac = kernelType == KernelType::PHS ? 3.0 : eps;
    ReconVectors RVs(kFac, core.SI.monoDeg[nS], KFVM_D_DECL(xs, ys, zs));

    // Find weights on west face
    auto wWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(wWts), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(mhalf, fq1, fq2), wWts);

    // Map weights from west face to other faces
    auto eWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry ewStenSym(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                              KFVM_D_DECL(half, fq1, fq2), KFVM_D_DECL(mhalf, fq1, fq2));
    ewStenSym.mapWeights(eWts, wWts);

    auto sWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry swStenSym(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                              KFVM_D_DECL(fq1, mhalf, fq2), KFVM_D_DECL(mhalf, fq1, fq2));
    swStenSym.mapWeights(sWts, wWts);

    auto nWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry nsStenSym(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                              KFVM_D_DECL(fq1, half, fq2), KFVM_D_DECL(fq1, mhalf, fq2));
    nsStenSym.mapWeights(nWts, sWts);

#if (SPACE_DIM == 3)
    auto bWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry bwStenSym(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                              KFVM_D_DECL(fq1, fq2, mhalf), KFVM_D_DECL(mhalf, fq1, fq2));
    bwStenSym.mapWeights(bWts, wWts);

    auto tWts =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry tbStenSym(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xs, ys, zs),
                              KFVM_D_DECL(fq1, fq2, half), KFVM_D_DECL(fq1, fq2, mhalf));
    tbStenSym.mapWeights(tWts, bWts);
#endif

    // Find weights for interior values
    auto cWts = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(cWts), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(cq1, cq2, cq3), cWts);

    // Find weights for each derivative type
    for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
      auto dWts = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
      // Note that first entry in derivSet is skipped since it is just a point value
      RVs.recVecs<decltype(dWts), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
          core.derivSet.deg[nI + 1], KFVM_D_DECL(0.0, 0.0, 0.0), dWts);
      double scl = std::pow(geom.dx, core.derivSet.deg[nI + 1][0]);
      scl *= std::pow(geom.dy, core.derivSet.deg[nI + 1][1]);
#if (SPACE_DIM == 3)
      scl *= std::pow(geom.dz, core.derivSet.deg[nI + 1][2]);
#endif
      for (idx_t nC = 0; nC < subSize; nC++) {
        dWts(nC) *= scl;
      }
    }
  }

  // Find weights for biased substencils
  {
    // First find west stencil
    idx_t nS = 2;

    // stencil as double arrays
    idx_t subSize = core.SI.nCellsBias;
    std::vector<double> KFVM_D_DECL(xw, yw, zw);
    off2Double(nS, subSize, KFVM_D_DECL(xw, yw, zw));

    // Reconstruction vector generator
    const double kFac = kernelType == KernelType::PHS ? 2.0 * STENCIL_RADIUS + 1.0 : eps;
    ReconVectors RVs(kFac, core.SI.monoDeg[nS], KFVM_D_DECL(xw, yw, zw));

    // Find weights on each face
    auto wFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(wFace_wSten), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(mhalf, fq1, fq2), wFace_wSten);

    auto eFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(eFace_wSten), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(half, fq1, fq2), eFace_wSten);

    auto sFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(sFace_wSten), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(fq1, mhalf, fq2), sFace_wSten);

    auto nFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry nsSym(false, KFVM_D_DECL(xw, yw, zw), KFVM_D_DECL(xw, yw, zw),
                          KFVM_D_DECL(fq1, half, fq2), KFVM_D_DECL(fq1, mhalf, fq2));
    nsSym.mapWeights(nFace_wSten, sFace_wSten);

#if (SPACE_DIM == 3)
    auto bFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry bsSym(false, KFVM_D_DECL(xw, yw, zw), KFVM_D_DECL(xw, yw, zw),
                          KFVM_D_DECL(fq1, fq2, mhalf), KFVM_D_DECL(fq1, mhalf, fq2));
    bsSym.mapWeights(bFace_wSten, sFace_wSten);

    auto tFace_wSten =
        Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
    StencilSymmetry tbSym(false, KFVM_D_DECL(xw, yw, zw), KFVM_D_DECL(xw, yw, zw),
                          KFVM_D_DECL(fq1, fq2, half), KFVM_D_DECL(fq1, fq2, mhalf));
    tbSym.mapWeights(tFace_wSten, bFace_wSten);
#endif

    // Find weights for interior values
    auto cWts_wSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
    RVs.recVecs<decltype(cWts_wSten), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
        pointSpec, KFVM_D_DECL(cq1, cq2, cq3), cWts_wSten);

    // Find weights for each derivative type
    for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
      auto dWts = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
      // Note that first entry in derivSet is skipped since it is just a point value
      RVs.recVecs<decltype(dWts), KFVM_D_DECL(NDeriv, NDeriv, NDeriv)>(
          core.derivSet.deg[nI + 1], KFVM_D_DECL(0.0, 0.0, 0.0), dWts);
      double scl = std::pow(geom.dx, core.derivSet.deg[nI + 1][0]);
      scl *= std::pow(geom.dy, core.derivSet.deg[nI + 1][1]);
#if (SPACE_DIM == 3)
      scl *= std::pow(geom.dz, core.derivSet.deg[nI + 1][2]);
#endif
      for (idx_t nC = 0; nC < subSize; nC++) {
        dWts(nC) *= scl;
      }
    }

    // Use west substencil to fill east substencil
    {
      nS = 3;

      // stencil as double arrays
      std::vector<double> KFVM_D_DECL(xs, ys, zs);
      off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

      auto wFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Wface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(mhalf, fq1, fq2),
                                  KFVM_D_DECL(half, fq1, fq2));
      ewSym_Wface.mapWeights(wFace_eSten, eFace_wSten);

      auto eFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Eface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(half, fq1, fq2),
                                  KFVM_D_DECL(mhalf, fq1, fq2));
      ewSym_Eface.mapWeights(eFace_eSten, wFace_wSten);

      auto sFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Sface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, mhalf, fq2),
                                  KFVM_D_DECL(fq1, mhalf, fq2));
      ewSym_Sface.mapWeights(sFace_eSten, sFace_wSten);

      auto nFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Nface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, half, fq2),
                                  KFVM_D_DECL(fq1, half, fq2));
      ewSym_Nface.mapWeights(nFace_eSten, nFace_wSten);

#if (SPACE_DIM == 3)
      auto bFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Bface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, mhalf),
                                  KFVM_D_DECL(fq1, fq2, mhalf));
      ewSym_Bface.mapWeights(bFace_eSten, bFace_wSten);

      auto tFace_eSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_Tface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, half),
                                  KFVM_D_DECL(fq1, fq2, half));
      ewSym_Tface.mapWeights(tFace_eSten, tFace_wSten);
#endif

      auto cWts_eSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry ewSym_cell(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                 KFVM_D_DECL(cq1, cq2, cq3), KFVM_D_DECL(cq1, cq2, cq3));
      ewSym_cell.mapWeights(cWts_eSten, cWts_wSten);

      // Map derivative weights
      StencilSymmetry ewSym_deriv(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(dq1, dq2, dq3), KFVM_D_DECL(dq1, dq2, dq3));
      for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
        auto dWts_wSten = Kokkos::subview(h_deriv, 2, nI, Kokkos::ALL);
        auto dWts_eSten = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
        ewSym_deriv.mapWeight(dWts_eSten, dWts_wSten);
      }
    }

    // Use west substencil to fill south substencil
    {
      nS = 4;

      // stencil as double arrays
      std::vector<double> KFVM_D_DECL(xs, ys, zs);
      off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

      auto wFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Wface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(mhalf, fq1, fq2),
                                  KFVM_D_DECL(fq1, half, fq2));
      swSym_Wface.mapWeights(wFace_sSten, nFace_wSten);

      auto eFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Eface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(half, fq1, fq2),
                                  KFVM_D_DECL(fq1, mhalf, fq2));
      swSym_Eface.mapWeights(eFace_sSten, sFace_wSten);

      auto sFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Sface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, mhalf, fq2),
                                  KFVM_D_DECL(mhalf, fq1, fq2));
      swSym_Sface.mapWeights(sFace_sSten, wFace_wSten);

      auto nFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Nface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, half, fq2),
                                  KFVM_D_DECL(half, fq1, fq2));
      swSym_Nface.mapWeights(nFace_sSten, eFace_wSten);

#if (SPACE_DIM == 3)
      auto bFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Bface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, mhalf),
                                  KFVM_D_DECL(fq1, fq2, mhalf));
      swSym_Bface.mapWeights(bFace_sSten, bFace_wSten);

      auto tFace_sSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_Tface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, half),
                                  KFVM_D_DECL(fq1, fq2, half));
      swSym_Tface.mapWeights(tFace_sSten, tFace_wSten);
#endif

      auto cWts_sSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry swSym_cell(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                 KFVM_D_DECL(cq1, cq2, cq3), KFVM_D_DECL(cq1, cq2, cq3));
      swSym_cell.mapWeights(cWts_sSten, cWts_wSten);

      // Map derivative weights
      StencilSymmetry swSym_deriv(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(dq1, dq2, dq3), KFVM_D_DECL(dq1, dq2, dq3));
      for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
        auto dWts_wSten = Kokkos::subview(h_deriv, 2, nI, Kokkos::ALL);
        auto dWts_sSten = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
        swSym_deriv.mapWeight(dWts_sSten, dWts_wSten);
      }
    }

    // Use west substencil to fill north substencil
    {
      nS = 5;

      // stencil as double arrays
      std::vector<double> KFVM_D_DECL(xs, ys, zs);
      off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

      auto wFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Wface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(mhalf, fq1, fq2),
                                  KFVM_D_DECL(fq1, mhalf, fq2));
      nwSym_Wface.mapWeights(wFace_nSten, sFace_wSten);

      auto eFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Eface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(half, fq1, fq2),
                                  KFVM_D_DECL(fq1, half, fq2));
      nwSym_Eface.mapWeights(eFace_nSten, nFace_wSten);

      auto sFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Sface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, mhalf, fq2),
                                  KFVM_D_DECL(half, fq1, fq2));
      nwSym_Sface.mapWeights(sFace_nSten, eFace_wSten);

      auto nFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Nface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, half, fq2),
                                  KFVM_D_DECL(mhalf, fq1, fq2));
      nwSym_Nface.mapWeights(nFace_nSten, wFace_wSten);

#if (SPACE_DIM == 3)
      auto bFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Bface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, mhalf),
                                  KFVM_D_DECL(fq1, fq2, mhalf));
      nwSym_Bface.mapWeights(bFace_nSten, bFace_wSten);

      auto tFace_nSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_Tface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, half),
                                  KFVM_D_DECL(fq1, fq2, half));
      nwSym_Tface.mapWeights(tFace_nSten, tFace_wSten);
#endif

      auto cWts_nSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry nwSym_cell(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                 KFVM_D_DECL(cq1, cq2, cq3), KFVM_D_DECL(cq1, cq2, cq3));
      nwSym_cell.mapWeights(cWts_nSten, cWts_wSten);

      // Map derivative weights
      StencilSymmetry nwSym_deriv(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(dq1, dq2, dq3), KFVM_D_DECL(dq1, dq2, dq3));
      for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
        auto dWts_wSten = Kokkos::subview(h_deriv, 2, nI, Kokkos::ALL);
        auto dWts_nSten = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
        nwSym_deriv.mapWeight(dWts_nSten, dWts_wSten);
      }
    }

#if (SPACE_DIM == 3)
    // Use west substencil to fill bottom substencil
    {
      nS = 6;

      // stencil as double arrays
      std::vector<double> KFVM_D_DECL(xs, ys, zs);
      off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

      auto wFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Wface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(mhalf, fq1, fq2),
                                  KFVM_D_DECL(fq1, fq2, half));
      bwSym_Wface.mapWeights(wFace_bSten, tFace_wSten);

      auto eFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Eface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(half, fq1, fq2),
                                  KFVM_D_DECL(fq1, fq2, mhalf));
      bwSym_Eface.mapWeights(eFace_bSten, bFace_wSten);

      auto sFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Sface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, mhalf, fq2),
                                  KFVM_D_DECL(fq1, mhalf, fq2));
      bwSym_Sface.mapWeights(sFace_bSten, sFace_wSten);

      auto nFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Nface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, half, fq2),
                                  KFVM_D_DECL(fq1, half, fq2));
      bwSym_Nface.mapWeights(nFace_bSten, nFace_wSten);

      auto bFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Bface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, mhalf),
                                  KFVM_D_DECL(mhalf, fq1, fq2));
      bwSym_Bface.mapWeights(bFace_bSten, wFace_wSten);

      auto tFace_bSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_Tface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, half),
                                  KFVM_D_DECL(half, fq1, fq2));
      bwSym_Tface.mapWeights(tFace_bSten, eFace_wSten);

      auto cWts_bSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry bwSym_cell(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                 KFVM_D_DECL(cq1, cq2, cq3), KFVM_D_DECL(cq1, cq2, cq3));
      bwSym_cell.mapWeights(cWts_bSten, cWts_wSten);

      // Map derivative weights
      StencilSymmetry bwSym_deriv(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(dq1, dq2, dq3), KFVM_D_DECL(dq1, dq2, dq3));
      for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
        auto dWts_wSten = Kokkos::subview(h_deriv, 2, nI, Kokkos::ALL);
        auto dWts_bSten = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
        bwSym_deriv.mapWeight(dWts_bSten, dWts_wSten);
      }
    }

    // Use west substencil to fill top substencil
    {
      nS = 7;

      // stencil as double arrays
      std::vector<double> KFVM_D_DECL(xs, ys, zs);
      off2Double(nS, subSize, KFVM_D_DECL(xs, ys, zs));

      auto wFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::west), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Wface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(mhalf, fq1, fq2),
                                  KFVM_D_DECL(fq1, fq2, mhalf));
      twSym_Wface.mapWeights(wFace_tSten, bFace_wSten);

      auto eFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::east), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Eface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(half, fq1, fq2),
                                  KFVM_D_DECL(fq1, fq2, half));
      twSym_Eface.mapWeights(eFace_tSten, tFace_wSten);

      auto sFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::south), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Sface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, mhalf, fq2),
                                  KFVM_D_DECL(fq1, mhalf, fq2));
      twSym_Sface.mapWeights(sFace_tSten, sFace_wSten);

      auto nFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::north), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Nface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, half, fq2),
                                  KFVM_D_DECL(fq1, half, fq2));
      twSym_Nface.mapWeights(nFace_tSten, nFace_wSten);

      auto bFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::bottom), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Bface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, mhalf),
                                  KFVM_D_DECL(half, fq1, fq2));
      twSym_Bface.mapWeights(bFace_tSten, eFace_wSten);

      auto tFace_tSten =
          Kokkos::subview(h_face, nS, idx_t(FaceLabel::top), Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_Tface(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(fq1, fq2, half),
                                  KFVM_D_DECL(mhalf, fq1, fq2));
      twSym_Tface.mapWeights(tFace_tSten, wFace_wSten);

      auto cWts_tSten = Kokkos::subview(h_cellF, nS, Kokkos::ALL, Kokkos::ALL);
      StencilSymmetry twSym_cell(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                 KFVM_D_DECL(cq1, cq2, cq3), KFVM_D_DECL(cq1, cq2, cq3));
      twSym_cell.mapWeights(cWts_tSten, cWts_wSten);

      // Map derivative weights
      StencilSymmetry twSym_deriv(false, KFVM_D_DECL(xs, ys, zs), KFVM_D_DECL(xw, yw, zw),
                                  KFVM_D_DECL(dq1, dq2, dq3), KFVM_D_DECL(dq1, dq2, dq3));
      for (idx_t nI = 0; nI < Core::SI.nIndic; nI++) {
        auto dWts_wSten = Kokkos::subview(h_deriv, 2, nI, Kokkos::ALL);
        auto dWts_tSten = Kokkos::subview(h_deriv, nS, nI, Kokkos::ALL);
        twSym_deriv.mapWeight(dWts_tSten, dWts_wSten);
      }
    }
#endif
  }

  // Unflatten cell quadrature weights
  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
  for (int nS = 0; nS < core.SI.nSub; nS++) {
    for (int nQ3 = 0; nQ3 < nzQuad; nQ3++) {
      for (int nQ2 = 0; nQ2 < NUM_QUAD_PTS; nQ2++) {
        for (int nQ1 = 0; nQ1 < NUM_QUAD_PTS; nQ1++) {
          idx_t idx = nQ1 * core.SI.nqFace * nzQuad + nQ2 * nzQuad + nQ3;
          for (int nC = 0; nC < core.SI.nCellsFull; nC++) {
            h_cell(nS, KFVM_D_DECL(nQ1, nQ2, nQ3), nC) = h_cellF(nS, idx, nC);
          }
        }
      }
    }
  }

  // Copy to views on the right memory space
  Kokkos::deep_copy(faceWeights, h_face);
  Kokkos::deep_copy(cellWeights, h_cell);
  Kokkos::deep_copy(derivWeights, h_deriv);
}

} // end namespace Stencil

} // end namespace KFVM
