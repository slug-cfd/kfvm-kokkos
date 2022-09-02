#ifndef STENCIL_H_
#define STENCIL_H_

#include <Kokkos_View.hpp>

#include <cstdint>
#include <cstdio>

#include "../SimVar.H"
#include "../Types.H"
#include "../ProblemSetup.H"

namespace KFVM {

  namespace Stencil {
    
// Cell counts for various arrays in Stencil indexed by radius
// This makes radius an adjustable (compile time) parameter
// but also allows all arrays to remain static
// Radius 0 and 1 ill defined.
    template <int dim,idx_t rad> struct StencilInfo;

    // 2D specializations
    // Radius 2
    template<> struct StencilInfo<2,2>
    {
      // Number of face quadrature points per dimension
      static constexpr idx_t nqFace = NUM_QUAD_PTS;
      // Total after tensoring across dimensions
      static constexpr idx_t nqFace_d = nqFace;
      // Number of interior quadrature points per dimension
      static constexpr idx_t nqCell = 2;
      // Total after tensoring across dimensions
      static constexpr idx_t nqCell_d = nqCell*nqCell;
      // Total number of cells in full stencil
      static constexpr idx_t nCellsFull = 13;
      // Total number of cells in central substencil
      static constexpr idx_t nCellsCtr = 5;
      // Total number of cells in biased substencils
      static constexpr idx_t nCellsBias = 5;
      // Number of derivatives in smoothness indicator
      static constexpr idx_t nDeriv = 4;
      // Number of total smoothness indicator entries to form
      static constexpr idx_t nIndic = nDeriv*nqCell_d;
      // Total workspace size needed by recon. kernel
      static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
      // Number of substencils
      static constexpr idx_t nSub = 6;
      // Number of biased substencils
      static constexpr idx_t nBias = 4;
    };

    // radius 3
    template<> struct StencilInfo<2,3>
    {
      static constexpr idx_t nqFace = NUM_QUAD_PTS;
      static constexpr idx_t nqFace_d = nqFace;
      static constexpr idx_t nqCell = 2;
      static constexpr idx_t nqCell_d = nqCell*nqCell;
      static constexpr idx_t nCellsFull = 29;
      static constexpr idx_t nCellsCtr = 13;
      static constexpr idx_t nCellsBias = 10;
      static constexpr idx_t nDeriv = 4;
      static constexpr idx_t nIndic = nDeriv*nqCell_d;
      static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
      static constexpr idx_t nSub = 6;
      static constexpr idx_t nBias = 4;
    };

    // 3D specializations
    // radius 2
    template<> struct StencilInfo<3,2>
    {
      static constexpr idx_t nqFace = NUM_QUAD_PTS;
      static constexpr idx_t nqFace_d = nqFace*nqFace;
      static constexpr idx_t nqCell = 2;
      static constexpr idx_t nqCell_d = nqCell*nqCell*nqCell;
      static constexpr idx_t nCellsFull = 33;
      static constexpr idx_t nCellsCtr = 7;
      static constexpr idx_t nCellsBias = 11;
      static constexpr idx_t nDeriv = 6;
      static constexpr idx_t nIndic = nDeriv*nqCell_d;
      static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
      static constexpr idx_t nSub = 8;
      static constexpr idx_t nBias = 6;
    };

    // radius 3
    template<> struct StencilInfo<3,3>
    {
      static constexpr idx_t nqFace = NUM_QUAD_PTS;
      static constexpr idx_t nqFace_d = nqFace*nqFace;
      static constexpr idx_t nqCell = 2;
      static constexpr idx_t nqCell_d = nqCell*nqCell*nqCell;
      static constexpr idx_t nCellsFull = 123;
      static constexpr idx_t nCellsCtr = 33;
      static constexpr idx_t nCellsBias = 32;
      static constexpr idx_t nDeriv = 6;
      static constexpr idx_t nIndic = nDeriv*nqCell_d;
      static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
      static constexpr idx_t nSub = 8;
      static constexpr idx_t nBias = 6;
    };

    // Set view types for stencil information and needed kernel workspace
    typedef Kokkos::View<idx_t[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nCellsFull],MemSpace> OffsetView;
    typedef Kokkos::View<idx_t[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nBias][StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nCellsBias],MemSpace> SubIdxView;
    typedef Kokkos::View<Real[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nSub][2*SPACE_DIM][StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nqFace_d][StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nCellsFull],MemSpace> FaceWtView;
    typedef Kokkos::View<Real[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nSub][StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nIndic][StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nCellsFull],MemSpace> CellWtView;
#if (SPACE_DIM == 2)
    typedef Kokkos::View<Real**[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nWorkspace][NUM_VARS],MemSpace> WorkView;
#else
    typedef Kokkos::View<Real***[StencilInfo<SPACE_DIM,STENCIL_RADIUS>::nWorkspace][NUM_VARS],MemSpace> WorkView;
#endif

    struct Stencil
    {
      static constexpr idx_t rad = STENCIL_RADIUS;

      enum SubSten {full = 0,center,west,east,south,north,bottom,top};
      
      struct Core
      {
	// StencilInfo provides counts of all relevant quantities
        static constexpr StencilInfo<SPACE_DIM,rad> SI {};
	
	// Stencil layout and indexing
        // Offsets from (i,j,k) describing full stencil
        std::array<idx_t,SI.nCellsFull> lOff,tOff,ttOff;
        // Index maps into biased substencils
        std::array<std::array<idx_t,SI.nCellsBias>,SI.nBias> subIdx;
	
	Core() { findStencil(); }
      private:
	idx_t coord2idx(idx_t,idx_t);
	idx_t coord2idx(idx_t,idx_t,idx_t);
        void findStencil();
        void findSubStencils();
      };

      Core core;

      // Layout of cells in stencil, and distribution into substencils
      OffsetView lOff,tOff,ttOff;
      SubIdxView subIdx;

      // stencil weights for reconstruction on faces
      // indices are substencil index/face index/quad point on face/cell in stencil
      // weights on smaller substencils will be extended by zeros initially
      FaceWtView faceWeights;

      // stencil weights for derivative reconstruction in cell
      // indices are substencil index/derivative type and location/cell in stencil
      CellWtView derivWeights;

      Stencil(Real lfac_):
	core(),
	lOff("Stencil::lOff"),
	tOff("Stencil::tOff"),
	ttOff("Stencil::ttOff"),
	subIdx("Stencil::subIdx"),
	faceWeights("Stencil::faceWeights"),
	derivWeights("Stencil::derivWeights")
      {
	Kokkos::Profiling::pushRegion("Stencil::Stencil");

	// Copy cell layout from core into device-side views
	auto h_lOff = Kokkos::create_mirror(lOff);
	auto h_tOff = Kokkos::create_mirror(tOff);
	auto h_ttOff = Kokkos::create_mirror(ttOff);
	auto h_subIdx = Kokkos::create_mirror(subIdx);
	for (idx_t nC=0; nC<Core::SI.nCellsFull; nC++) {
	  h_lOff(nC) = core.lOff[nC];
	  h_tOff(nC) = core.tOff[nC];
	  h_ttOff(nC) = core.ttOff[nC];
	}
	for (idx_t nB=0; nB<Core::SI.nBias; nB++) {
	  for (idx_t nC=0; nC<Core::SI.nCellsBias; nC++) {
	    h_subIdx(nB,nC) = core.subIdx[nB][nC];
	  }
	}
	Kokkos::deep_copy(lOff,h_lOff);
	Kokkos::deep_copy(tOff,h_tOff);
	Kokkos::deep_copy(ttOff,h_ttOff);
	Kokkos::deep_copy(subIdx,h_subIdx);
	
	findWeights(static_cast<double>(lfac_));
	
	Kokkos::Profiling::popRegion();
      }

      private:
      // This should never be default or copy constructed
      Stencil() = delete;
      Stencil(const Stencil&) = delete;

      void findWeights(double);
    };

  } // end namespace Stencil

} // end namespace KFVM

#endif
