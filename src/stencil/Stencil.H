#ifndef STENCIL_H_
#define STENCIL_H_

#include <cstdint>
#include <cstdio>
#include <vector>

#include <Kokkos_Core.hpp>

#include "../Geometry.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "Definitions.H"
#include "StencilInfo.H"

namespace KFVM {

namespace Stencil {

// Set view types for stencil information and needed kernel workspace
typedef Kokkos::View<idx_t[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull], MemSpace>
    OffsetView;
typedef Kokkos::View<idx_t[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nBias]
                          [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsBias],
                     MemSpace>
    SubIdxView;
typedef Kokkos::View<Real[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nSub][2 * SPACE_DIM]
                         [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nqFace_d]
                         [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull],
                     MemSpace>
    FaceWtView;
#if (SPACE_DIM == 2)
typedef Kokkos::View<
    Real[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nSub][NUM_QUAD_PTS][NUM_QUAD_PTS]
        [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull],
    MemSpace>
    CellWtView;
#else
typedef Kokkos::View<
    Real[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nSub][NUM_QUAD_PTS][NUM_QUAD_PTS]
        [NUM_QUAD_PTS][StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull],
    MemSpace>
    CellWtView;
#endif
typedef Kokkos::View<Real[StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nSub]
                         [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nIndic]
                         [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nCellsFull],
                     MemSpace>
    DerivWtView;
typedef Kokkos::View<
    Real * [StencilInfo<SPACE_DIM, STENCIL_RADIUS>::nWorkspace][NUM_VARS], MemSpace>
    WorkView;

struct Stencil {
  static constexpr idx_t rad = STENCIL_RADIUS;

  enum SubSten { full = 0, center, west, east, south, north, bottom, top };

  struct Core {
    // StencilInfo provides counts of all relevant quantities
    static constexpr StencilInfo<SPACE_DIM, rad> SI{};

    // Monomial enumeration also gives all derivatives needed for smoothness indicators
    Monomials::MonomialEnum derivSet;

    // Stencil layout and indexing
    // Offsets from (i,j,k) describing full stencil
    std::array<idx_t, SI.nCellsFull> lOff, tOff, ttOff;

    // Index maps into biased substencils
    std::array<std::array<idx_t, SI.nCellsBias>, SI.nBias> subIdx;

    Core() : derivSet(SI.nDeriv) { findStencil(); }

   private:
    idx_t coord2idx(idx_t, idx_t);
    idx_t coord2idx(idx_t, idx_t, idx_t);
    void findStencil();
    void findSubStencils();
  };

  // Base information about the stencil sizes, number of quad pts etc
  Core core;

  // Geometry object for scaling derivatives correctly
  const Geometry &geom;

  // Layout of cells in stencil, and distribution into substencils
  OffsetView lOff, tOff, ttOff;
  SubIdxView subIdx;

  // stencil weights for reconstruction on faces
  // indices are substencil index/face index/quad point on face/cell in stencil
  // weights on smaller substencils will be extended by zeros initially
  FaceWtView faceWeights;

  // stencil weights for reconstructing interior cell values
  // indices are substencil index/x-quad/y-quad/(z-quad if 3d)/cell in stencil
  CellWtView cellWeights;

  // stencil weights for derivative reconstruction in cell
  // indices are substencil index/derivative type and location/cell in stencil
  DerivWtView derivWeights;

  Stencil(const Real lfac_, const Geometry &geom_)
      : core(), geom(geom_), lOff("Stencil::lOff"), tOff("Stencil::tOff"),
        ttOff("Stencil::ttOff"), subIdx("Stencil::subIdx"),
        faceWeights("Stencil::faceWeights"), cellWeights("Stencil::cellWeights"),
        derivWeights("Stencil::derivWeights") {
    Kokkos::Profiling::pushRegion("Stencil::Stencil");

    // Copy cell layout from core into device-side views
    auto h_lOff = Kokkos::create_mirror(lOff);
    auto h_tOff = Kokkos::create_mirror(tOff);
    auto h_ttOff = Kokkos::create_mirror(ttOff);
    auto h_subIdx = Kokkos::create_mirror(subIdx);
    for (idx_t nC = 0; nC < Core::SI.nCellsFull; nC++) {
      h_lOff(nC) = core.lOff[nC];
      h_tOff(nC) = core.tOff[nC];
      h_ttOff(nC) = core.ttOff[nC];
    }
    for (idx_t nB = 0; nB < Core::SI.nBias; nB++) {
      for (idx_t nC = 0; nC < Core::SI.nCellsBias; nC++) {
        h_subIdx(nB, nC) = core.subIdx[nB][nC];
      }
    }
    Kokkos::deep_copy(lOff, h_lOff);
    Kokkos::deep_copy(tOff, h_tOff);
    Kokkos::deep_copy(ttOff, h_ttOff);
    Kokkos::deep_copy(subIdx, h_subIdx);

    findWeights(static_cast<double>(lfac_));

    Kokkos::Profiling::popRegion();
  }

 private:
  // This should never be default or copy constructed
  Stencil() = delete;
  Stencil(const Stencil &) = delete;

  void off2Double(idx_t, idx_t,
                  KFVM_D_DECL(std::vector<double> &, std::vector<double> &,
                              std::vector<double> &));
  void findWeights(double);
};

} // end namespace Stencil

} // end namespace KFVM

#endif
