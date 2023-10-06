#ifndef STENCILINFO_H_
#define STENCILINFO_H_

#include "../Types.H"
#include "EvalFunctionals.H"
#include "Monomials.H"

namespace KFVM {

namespace Stencil {

// Cell counts for various arrays in Stencil indexed by radius
// This makes radius an adjustable (compile time) parameter
// but also allows all arrays to remain static
// Radius 0 and 1 ill defined.
template <int dim, idx_t rad>
struct StencilInfo;

// 2D specializations
// Radius 2
template <>
struct StencilInfo<2, 2> {
  // Number of face quadrature points per dimension
  static constexpr idx_t nqFace = NUM_QUAD_PTS;
  // Total on each face after tensoring across dimensions
  static constexpr idx_t nqFace_d = nqFace;
  // Total interior to cell for source terms
  static constexpr idx_t nqCell_d = nqFace * nqFace;
  // Total number of cells in full stencil
  static constexpr idx_t nCellsFull = 13;
  // Total number of cells in central substencil
  static constexpr idx_t nCellsCtr = 5;
  // Total number of cells in biased substencils
  static constexpr idx_t nCellsBias = 5;
  // Total degree of derivatives in smoothness indicator
  static constexpr idx_t nDeriv = 2;
  // Number of derivatives in smoothness indicator
  static constexpr idx_t nIndic = Monomials::NumMonos(nDeriv) - 1;
  // Total workspace size needed by recon. kernel
  static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
  // Number of substencils
  static constexpr idx_t nSub = 6;
  // Number of biased substencils
  static constexpr idx_t nBias = 4;
  // Max monomial degree for each substencil type
  static constexpr std::array<idx_t, 3> monoDeg{3, 1, 1};
};

// radius 3
template <>
struct StencilInfo<2, 3> {
  static constexpr idx_t nqFace = NUM_QUAD_PTS;
  static constexpr idx_t nqFace_d = nqFace;
  static constexpr idx_t nqCell_d = nqFace * nqFace;
  static constexpr idx_t nCellsFull = 29;
  static constexpr idx_t nCellsCtr = 13;
  static constexpr idx_t nCellsBias = 10;
  static constexpr idx_t nDeriv = 3;
  static constexpr idx_t nIndic = Monomials::NumMonos(nDeriv) - 1;
  static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
  static constexpr idx_t nSub = 6;
  static constexpr idx_t nBias = 4;
  static constexpr std::array<idx_t, 3> monoDeg{5, 3, 2};
};

// 3D specializations
// radius 2
template <>
struct StencilInfo<3, 2> {
  static constexpr idx_t nqFace = NUM_QUAD_PTS;
  static constexpr idx_t nqFace_d = nqFace * nqFace;
  static constexpr idx_t nqCell_d = nqFace * nqFace * nqFace;
  static constexpr idx_t nCellsFull = 33;
  static constexpr idx_t nCellsCtr = 7;
  static constexpr idx_t nCellsBias = 11;
  static constexpr idx_t nDeriv = 3;
  static constexpr idx_t nIndic = Monomials::NumMonos(nDeriv) - 1;
  static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
  static constexpr idx_t nSub = 8;
  static constexpr idx_t nBias = 6;
  static constexpr std::array<idx_t, 3> monoDeg{3, 1, 1};
};

// radius 3
template <>
struct StencilInfo<3, 3> {
  static constexpr idx_t nqFace = NUM_QUAD_PTS;
  static constexpr idx_t nqFace_d = nqFace * nqFace;
  static constexpr idx_t nqCell_d = nqFace * nqFace * nqFace;
  static constexpr idx_t nCellsFull = 123;
  static constexpr idx_t nCellsCtr = 33;
  static constexpr idx_t nCellsBias = 32;
  static constexpr idx_t nDeriv = 3;
  static constexpr idx_t nIndic = Monomials::NumMonos(nDeriv) - 1;
  static constexpr idx_t nWorkspace = nCellsFull + nCellsBias + nIndic;
  static constexpr idx_t nSub = 8;
  static constexpr idx_t nBias = 6;
  static constexpr std::array<idx_t, 3> monoDeg{5, 3, 2};
};

} // end namespace Stencil

} // end namespace KFVM

#endif
