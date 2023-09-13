#ifndef STENCIL_K_H_
#define STENCIL_K_H_

// File: Stencil_K.H
// Purpose: Define compute kernels for reconstructing Riemann states
//          from cell average data

#include <cstdint>

#include <Kokkos_Core.hpp>

#include <KokkosBatched_Gemv_Decl.hpp>
#include <KokkosBatched_Gemv_Serial_Impl.hpp>

#include <KokkosBatched_Gemm_Decl.hpp>
#include <KokkosBatched_Gemm_Serial_Impl.hpp>

#include <Definitions.H>

#include "../Dimension.H"
#include "../Geometry.H"
#include "../ParameterStructs.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "../physics/Physics_K.H"
#include "Stencil.H"

namespace KFVM {

namespace Stencil {

template <class UViewType>
struct KernelLinearRecon_K {
  UViewType U;
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  SourceDataView source;
  bool haveSource;
  OffsetView KFVM_D_DECL(lOff, tOff, ttOff);
  FaceWtView faceWeight;
  CellWtView cellWeight;
  const int nCellsFull;
  const int nqFace_d;

  KernelLinearRecon_K(const UViewType &U_,
                      KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                                  const FaceDataView &rsZ_),
                      const SourceDataView source_, bool haveSource_,
                      KFVM_D_DECL(const OffsetView &lOff_, const OffsetView &tOff_,
                                  const OffsetView &ttOff_),
                      const FaceWtView &face_, const CellWtView &cell_)
      : U(U_), KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)),
        KFVM_D_DECL(lOff(lOff_), tOff(tOff_), ttOff(ttOff_)), source(source_),
        haveSource(haveSource_), faceWeight(face_), cellWeight(cell_),
        nCellsFull(Stencil::Core::SI.nCellsFull), nqFace_d(Stencil::Core::SI.nqFace_d) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Pull Riemann states subviews out into array
#if (SPACE_DIM == 2)
    decltype(Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#else
    decltype(Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#endif

    // Fill in Riemann states
    for (int nF = 0; nF < 2 * SPACE_DIM; nF++) {
      auto rWtsF = Kokkos::subview(faceWeight, 0, nF, Kokkos::ALL, Kokkos::ALL);
      for (int nQ = 0; nQ < nqFace_d; nQ++) {
        for (int nV = 0; nV < NUM_VARS; nV++) {
          // Zero out state
          rsFace[nF](nQ, nV) = Real(0.0);
          // Apply stencil weights
          for (int nC = 0; nC < nCellsFull; nC++) {
            rsFace[nF](nQ, nV) +=
                rWtsF(nQ, nC) *
                U(KFVM_D_DECL(i + lOff(nC), j + tOff(nC), k + ttOff(nC)), nV);
          }
        }
      }
    }

    // Fill in source terms
    if (haveSource) {
      const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      auto cellVals = Kokkos::subview(source, KFVM_D_DECL(i, j, k),
                                      KFVM_D_DECL(Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                                      Kokkos::ALL);

      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            auto state = Kokkos::subview(cellVals, KFVM_D_DECL(nQ, nR, nS), Kokkos::ALL);
            auto rWtsF =
                Kokkos::subview(cellWeight, 0, KFVM_D_DECL(nQ, nR, nS), Kokkos::ALL);
            for (int nV = 0; nV < NUM_VARS; nV++) {
              // Zero out state
              state(nV) = Real(0.0);
              // Apply stencil weights
              for (int nC = 0; nC < nCellsFull; nC++) {
                state(nV) +=
                    rWtsF(nC) *
                    U(KFVM_D_DECL(i + lOff(nC), j + tOff(nC), k + ttOff(nC)), nV);
              }
            }
          }
        }
      }
    }
  }
};

template <class UViewType>
struct KernelWenoRecon_K {
  UViewType U;
  idx_t nX, nY, nZ;
  idx_t tX, tY, tZ;
  FaceDataView KFVM_D_DECL(rsX, rsY, rsZ);
  SourceDataView source;
  bool haveSource;
  WorkView Work;
  FlagMap wenoMap;
  OffsetView KFVM_D_DECL(lOff, tOff, ttOff);
  SubIdxView subIdx;
  FaceWtView faceWeight;
  CellWtView cellWeight;
  DerivWtView derivWeight;
  EosParameters eosParams;
  const int nCellsFull, nCellsCtr, nCellsBias;
  const int nSub, nBias, nIndic, nqFace_d;
  const Real gHi, gLo, eps;

  KernelWenoRecon_K(const UViewType &U_, KFVM_D_DECL(idx_t nX_, idx_t nY_, idx_t nZ_),
                    KFVM_D_DECL(idx_t tX_, idx_t tY_, idx_t tZ_),
                    KFVM_D_DECL(const FaceDataView &rsX_, const FaceDataView &rsY_,
                                const FaceDataView &rsZ_),
                    const SourceDataView source_, bool haveSource_, const WorkView &Work_,
                    const FlagMap &wenoMap_,
                    KFVM_D_DECL(const OffsetView &lOff_, const OffsetView &tOff_,
                                const OffsetView &ttOff_),
                    const SubIdxView &subIdx_, const FaceWtView &face_,
                    const CellWtView &cell_, const DerivWtView &deriv_,
                    const EosParameters &eosParams_)
      : U(U_), KFVM_D_DECL(nX(nX_), nY(nY_), nZ(nZ_)),
        KFVM_D_DECL(tX(tX_), tY(tY_), tZ(tZ_)),
        KFVM_D_DECL(rsX(rsX_), rsY(rsY_), rsZ(rsZ_)), source(source_),
        haveSource(haveSource_), Work(Work_), wenoMap(wenoMap_),
        KFVM_D_DECL(lOff(lOff_), tOff(tOff_), ttOff(ttOff_)), subIdx(subIdx_),
        faceWeight(face_), cellWeight(cell_), derivWeight(deriv_), eosParams(eosParams_),
        nCellsFull(Stencil::Core::SI.nCellsFull), nCellsCtr(Stencil::Core::SI.nCellsCtr),
        nCellsBias(Stencil::Core::SI.nCellsBias), nSub(Stencil::Core::SI.nSub),
        nBias(Stencil::Core::SI.nBias), nIndic(Stencil::Core::SI.nIndic),
        nqFace_d(Stencil::Core::SI.nqFace_d),
#if (SPACE_DIM == 2)
        gHi(0.75), gLo(0.75), eps(1.e-40)
#else
        gHi(0.7), gLo(0.75), eps(1.e-40)
#endif
  {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    // Create subview slices of workspace for substencils
#if (SPACE_DIM == 2)
    // Trim i,j,k to local tile then lex to get workspace slot
    idx_t workIdx = tX * (j % tY) + (i % tX);
#else
    idx_t workIdx = tX * tY * (k % tZ) + tX * (j % tY) + (i % tX);
#endif
    auto Vf = Kokkos::subview(Work, workIdx, Kokkos::pair<int, int>(0, nCellsFull),
                              Kokkos::ALL);
    auto Vc =
        Kokkos::subview(Work, workIdx, Kokkos::pair<int, int>(0, nCellsCtr), Kokkos::ALL);
    auto Vb = Kokkos::subview(Work, workIdx,
                              Kokkos::pair<int, int>(nCellsFull, nCellsFull + nCellsBias),
                              Kokkos::ALL);
    // Subview slice to store derivatives in before accumulating into beta
    auto B = Kokkos::subview(
        Work, workIdx,
        Kokkos::make_pair(nCellsFull + nCellsBias, nCellsFull + nCellsBias + nIndic),
        Kokkos::ALL);

    // Pull Riemann state subviews out into array
#if (SPACE_DIM == 2)
    decltype(Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#else
    decltype(Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#endif

    // Linear weights
#if (SPACE_DIM == 2)
    const Real gamma[] = {gHi,
                          (Real(1.0) - gHi) * gLo,
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0)};
#else
    const Real gamma[] = {gHi,
                          (Real(1.0) - gHi) * gLo,
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0)};
#endif
    // Storage for nonlinear weights
    Real omega[Stencil::Core::SI.nSub][NUM_VARS];

    // Storage for reference state
    Real refState[NUM_VARS];

    // Save reference state
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Variable conversion context
    Physics::VarConvert<eqType> varConv(refState, eosParams.gamma);

    // Populate stencil values
    for (int nC = 0; nC < nCellsFull; nC++) {
      auto Uijk = Kokkos::subview(
          U, KFVM_D_DECL(i + lOff(nC), j + tOff(nC), k + ttOff(nC)), Kokkos::ALL);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        Vf(nC, nV) = Uijk(nV);
      }
    }

    for (int nC = 0; nC < nCellsFull; nC++) {
      auto state = Kokkos::subview(Vf, nC, Kokkos::ALL);
      varConv.cons2LinPrim(state);
    }

    // Find all smoothness indicators
    findSmoothness(Vf, Vc, Vb, B, omega);

    // Convert smoothness indicators to omegas
    createOmegas(omega, gamma);

    // Fill in Riemann states one face at a time
    fillRiemannStates(Vf, Vc, Vb, omega, rsFace);

    // Convert back to conservative variables
    for (int nF = 0; nF < 2 * SPACE_DIM; nF++) {
      for (int nQ = 0; nQ < nqFace_d; nQ++) {
        auto state = Kokkos::subview(rsFace[nF], nQ, Kokkos::ALL);
        varConv.linPrim2Cons(state);
      }
    }

    // Fill interior values if needed
    if (haveSource) {
      const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      auto cellVals = Kokkos::subview(source, KFVM_D_DECL(i, j, k),
                                      KFVM_D_DECL(Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                                      Kokkos::ALL);
      fillInterior(Vf, Vc, Vb, omega, cellVals);

      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            auto state = Kokkos::subview(cellVals, KFVM_D_DECL(nQ, nR, nS), Kokkos::ALL);
            varConv.linPrim2Cons(state);
          }
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(uint32_t workIdx) const {
    // bail out if this idx is invalid
    if (!wenoMap.valid_at(workIdx)) {
      return;
    }

    // convert key to coordinate
    idx_t key = wenoMap.key_at(workIdx);
#if (SPACE_DIM == 2)
    idx_t j = key / nX;
    idx_t i = key - nX * j;
    assert(key == nX * j + i);
#else
    idx_t k = key / (nX * nY);
    idx_t j = (key - nX * nY * k) / nX;
    idx_t i = key - nX * nY * k - nX * j;
    assert(key == nX * nY * k + nX * j + i);
#endif
    auto Vf = Kokkos::subview(Work, workIdx, Kokkos::pair<int, int>(0, nCellsFull),
                              Kokkos::ALL);
    auto Vc =
        Kokkos::subview(Work, workIdx, Kokkos::pair<int, int>(0, nCellsCtr), Kokkos::ALL);
    auto Vb = Kokkos::subview(Work, workIdx,
                              Kokkos::pair<int, int>(nCellsFull, nCellsFull + nCellsBias),
                              Kokkos::ALL);
    // Subview slice to store derivatives in before accumulating into beta
    auto B = Kokkos::subview(
        Work, workIdx,
        Kokkos::make_pair(nCellsFull + nCellsBias, nCellsFull + nCellsBias + nIndic),
        Kokkos::ALL);

    // Pull Riemann state subviews out into array
#if (SPACE_DIM == 2)
    decltype(Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#else
    decltype(Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL)) rsFace[] = {
        Kokkos::subview(rsX, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsX, i + 1, j, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsY, i, j + 1, k, 0, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k, 1, Kokkos::ALL, Kokkos::ALL),
        Kokkos::subview(rsZ, i, j, k + 1, 0, Kokkos::ALL, Kokkos::ALL)};
#endif

    // Linear weights
#if (SPACE_DIM == 2)
    const Real gamma[] = {gHi,
                          (Real(1.0) - gHi) * gLo,
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(4.0)};
#else
    const Real gamma[] = {gHi,
                          (Real(1.0) - gHi) * gLo,
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0),
                          (Real(1.0) - gHi) * (Real(1.0) - gLo) / Real(6.0)};
#endif
    // Storage for nonlinear weights
    Real omega[Stencil::Core::SI.nSub][NUM_VARS];

    // Storage for reference state
    Real refState[NUM_VARS];

    // Save reference state
    for (int nV = 0; nV < NUM_VARS; nV++) {
      refState[nV] = U(KFVM_D_DECL(i, j, k), nV);
    }

    // Variable conversion context
    Physics::VarConvert<eqType> varConv(refState, eosParams.gamma);

    // Populate stencil values
    for (int nC = 0; nC < nCellsFull; nC++) {
      auto Uijk = Kokkos::subview(
          U, KFVM_D_DECL(i + lOff(nC), j + tOff(nC), k + ttOff(nC)), Kokkos::ALL);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        Vf(nC, nV) = Uijk(nV);
      }
    }

    for (int nC = 0; nC < nCellsFull; nC++) {
      auto state = Kokkos::subview(Vf, nC, Kokkos::ALL);
      varConv.cons2LinPrim(state);
    }

    // Find all smoothness indicators
    findSmoothness(Vf, Vc, Vb, B, omega);

    // Convert smoothness indicators to omegas
    createOmegas(omega, gamma);

    // Fill in Riemann states one face at a time
    fillRiemannStates(Vf, Vc, Vb, omega, rsFace);

    // Convert back to conservative variables
    for (int nF = 0; nF < 2 * SPACE_DIM; nF++) {
      for (int nQ = 0; nQ < nqFace_d; nQ++) {
        auto state = Kokkos::subview(rsFace[nF], nQ, Kokkos::ALL);
        varConv.linPrim2Cons(state);
      }
    }

    // Fill interior values if needed
    if (haveSource) {
      const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
      auto cellVals = Kokkos::subview(source, KFVM_D_DECL(i, j, k),
                                      KFVM_D_DECL(Kokkos::ALL, Kokkos::ALL, Kokkos::ALL),
                                      Kokkos::ALL);
      fillInterior(Vf, Vc, Vb, omega, cellVals);

      for (int nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
        for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
          for (int nS = 0; nS < nzQuad; nS++) {
            auto state = Kokkos::subview(cellVals, KFVM_D_DECL(nQ, nR, nS), Kokkos::ALL);
            varConv.linPrim2Cons(state);
          }
        }
      }
    }
  }

  template <class ValsFullType, class ValsCtrType, class ValsBiasType, class RSArrayType>
  KOKKOS_INLINE_FUNCTION void
  fillRiemannStates(ValsFullType Vf, ValsCtrType Vc, ValsBiasType Vb,
                    Real omega[Stencil::Core::SI.nSub][NUM_VARS],
                    RSArrayType rsFace) const {
    using namespace KokkosBatched;

    // First apply full and center stencils
    for (int nF = 0; nF < 2 * SPACE_DIM; nF++) {
      auto rWtsF = Kokkos::subview(faceWeight, 0, nF, Kokkos::ALL, Kokkos::ALL);
      auto rWtsC = Kokkos::subview(faceWeight, 1, nF, Kokkos::ALL,
                                   Kokkos::make_pair(0, nCellsCtr));
      for (int nV = 0; nV < NUM_VARS; nV++) {
        auto rsFV = Kokkos::subview(rsFace[nF], Kokkos::ALL, nV);
        auto vf = Kokkos::subview(Vf, Kokkos::ALL, nV);
        auto vc = Kokkos::subview(Vc, Kokkos::ALL, nV);
        SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
            omega[0][nV], rWtsF, vf, Real(0.0), rsFV);
        SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
            omega[1][nV], rWtsC, vc, Real(1.0), rsFV);
      }
    }

    // Then apply biased stencils
    // Do these separately to minimize calls to fillbiased
    for (int nB = 0; nB < nBias; nB++) {
      fillBiased(nB, Vf, Vb);
      for (int nF = 0; nF < 2 * SPACE_DIM; nF++) {
        auto rWtsB = Kokkos::subview(faceWeight, 2 + nB, nF, Kokkos::ALL,
                                     Kokkos::make_pair(0, nCellsBias));
        for (int nV = 0; nV < NUM_VARS; nV++) {
          auto rsFV = Kokkos::subview(rsFace[nF], Kokkos::ALL, nV);
          auto vb = Kokkos::subview(Vb, Kokkos::ALL, nV);
          SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
              omega[2 + nB][nV], rWtsB, vb, Real(1.0), rsFV);
        }
      }
    }
  }

  template <class ValsFullType, class ValsCtrType, class ValsBiasType,
            class CellArrayType>
  KOKKOS_INLINE_FUNCTION void fillInterior(ValsFullType Vf, ValsCtrType Vc,
                                           ValsBiasType Vb,
                                           Real omega[Stencil::Core::SI.nSub][NUM_VARS],
                                           CellArrayType cellVals) const {
    using namespace KokkosBatched;

    const int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;

    // First apply full and center stencils
    // Compute states pencil-by-pencil for each fixed y,z pair
    for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
      for (int nS = 0; nS < nzQuad; nS++) {
        auto rWtsF =
            Kokkos::subview(cellWeight, 0, KFVM_D_DECL(Kokkos::ALL, nR, nS), Kokkos::ALL);
        auto rWtsC = Kokkos::subview(cellWeight, 1, KFVM_D_DECL(Kokkos::ALL, nR, nS),
                                     Kokkos::make_pair(0, nCellsCtr));
        for (int nV = 0; nV < NUM_VARS; nV++) {
          auto penV = Kokkos::subview(cellVals, KFVM_D_DECL(Kokkos::ALL, nR, nS), nV);
          auto vf = Kokkos::subview(Vf, Kokkos::ALL, nV);
          auto vc = Kokkos::subview(Vc, Kokkos::ALL, nV);
          SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
              omega[0][nV], rWtsF, vf, Real(0.0), penV);
          SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
              omega[1][nV], rWtsC, vc, Real(1.0), penV);
        }
      }
    }

    // Then apply biased stencils
    // Do these separately to minimize calls to fillbiased
    for (int nB = 0; nB < nBias; nB++) {
      fillBiased(nB, Vf, Vb);
      for (int nR = 0; nR < NUM_QUAD_PTS; nR++) {
        for (int nS = 0; nS < nzQuad; nS++) {
          auto rWtsB =
              Kokkos::subview(cellWeight, 2 + nB, KFVM_D_DECL(Kokkos::ALL, nR, nS),
                              Kokkos::make_pair(0, nCellsBias));
          for (int nV = 0; nV < NUM_VARS; nV++) {
            auto penV = Kokkos::subview(cellVals, KFVM_D_DECL(Kokkos::ALL, nR, nS), nV);
            auto vb = Kokkos::subview(Vb, Kokkos::ALL, nV);
            SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
                omega[2 + nB][nV], rWtsB, vb, Real(1.0), penV);
          }
        }
      }
    }
  }

  template <class ValsFullType, class ValsCtrType, class ValsBiasType,
            class DerivAccumType>
  KOKKOS_INLINE_FUNCTION void
  findSmoothness(ValsFullType Vf, ValsCtrType Vc, ValsBiasType Vb, DerivAccumType B,
                 Real omega[Stencil::Core::SI.nSub][NUM_VARS]) const {
    using namespace KokkosBatched;

    // Full stencil
    {
      auto dWts = Kokkos::subview(derivWeight, 0, Kokkos::ALL, Kokkos::ALL);
      SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(
          Real(1.0), dWts, Vf, Real(0.0), B);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        omega[0][nV] = Real(0.0);
        for (int nI = 0; nI < nIndic; nI++) {
          omega[0][nV] += B(nI, nV) * B(nI, nV);
        }
      }
    }

    // Central substencil
    {
      auto dWts =
          Kokkos::subview(derivWeight, 1, Kokkos::ALL, Kokkos::make_pair(0, nCellsCtr));
      SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(
          Real(1.0), dWts, Vc, Real(0.0), B);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        omega[1][nV] = Real(0.0);
        for (int nI = 0; nI < nIndic; nI++) {
          omega[1][nV] += B(nI, nV) * B(nI, nV);
        }
      }
    }

    // biased substencils
    for (int nB = 0; nB < nBias; nB++) {
      fillBiased(nB, Vf, Vb);
      auto dWts = Kokkos::subview(derivWeight, nB + 2, Kokkos::ALL,
                                  Kokkos::make_pair(0, nCellsBias));
      SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(
          Real(1.0), dWts, Vb, Real(0.0), B);
      for (int nV = 0; nV < NUM_VARS; nV++) {
        omega[nB + 2][nV] = Real(0.0);
        for (int nI = 0; nI < nIndic; nI++) {
          omega[nB + 2][nV] += B(nI, nV) * B(nI, nV);
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void createOmegas(Real omega[Stencil::Core::SI.nSub][NUM_VARS],
                    const Real gamma[Stencil::Core::SI.nSub]) const {
    using Kokkos::fabs;
    using Kokkos::pow;

    const int p = 2;

    for (int nV = 0; nV < NUM_VARS; nV++) {
      Real tot = Real(0.0);
      for (int nS = 0; nS < nSub; nS++) {
        omega[nS][nV] = gamma[nS] / (pow(omega[nS][nV], p) + eps);
        tot += omega[nS][nV];
      }

      // normalize them
      for (int nS = 0; nS < nSub; nS++) {
        omega[nS][nV] /= tot;
      }
    }

    // convert omegas to Weno-AO style weights
    for (int nV = 0; nV < NUM_VARS; nV++) {
      for (int nS = 1; nS < nSub; nS++) {
        omega[nS][nV] -= omega[0][nV] * gamma[nS] / gamma[0];
      }
      omega[0][nV] /= gamma[0];
    }
  }

  template <class ValsFullType, class ValsBiasType>
  KOKKOS_INLINE_FUNCTION void fillBiased(int nB, ValsFullType Vf, ValsBiasType Vb) const {
    for (int nV = 0; nV < NUM_VARS; nV++) {
      for (int nC = 0; nC < nCellsBias; nC++) {
        Vb(nC, nV) = Vf(subIdx(nB, nC), nV);
      }
    }
  }
};

} // namespace Stencil

} // end namespace KFVM

#endif
