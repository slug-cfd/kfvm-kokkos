#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include <Kokkos_Core.hpp>

#include <Definitions.H>
#include <cstdint>
#include <cstdio>

#include "BoundaryData.H"
#include "Dimension.H"
#include "Geometry.H"
#include "ProblemSetup.H"
#include "Types.H"
#include "io/WriterPDI.H"
#include "numeric/Numeric.H"
#include "physics/Physics_K.H"
#include "stencil/Stencil.H"

namespace KFVM {

// struct to manage linear vs weno reconstruction
struct WenoSelector {
  const ProblemSetup &ps; // Reference to outer ps for ease
  WenoFlagView wenoFlagView;
  FlagMap wenoFlagMap;
  Stencil::WorkView stenWork;
  int ntX, ntY, ntZ; // Tiles in x/y/z for full weno
  idx_t tX, tY, tZ;  // Tile size
  uint32_t tiledSize, currSize, nWeno;
  WenoSelector(const ProblemSetup &ps);
  void allocateTiled();
  void rehash(const uint32_t);
  template <class UViewType>
  void update(UViewType, KFVM_D_DECL(FaceDataView, FaceDataView, FaceDataView),
              const Real);
  auto tileRange(int i, int j, int k) const {
    idx_t xUp = i == (ntX - 1) ? ps.nX : (i + 1) * tX;
    idx_t yUp = j == (ntY - 1) ? ps.nY : (j + 1) * tY;
#if (SPACE_DIM == 3)
    idx_t zUp = k == (ntZ - 1) ? ps.nZ : (k + 1) * tZ;
#endif
    return Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<SPACE_DIM>,
                                 Kokkos::IndexType<idx_t>>(
        {KFVM_D_DECL(i * tX, j * tY, k * tZ)}, {KFVM_D_DECL(xUp, yUp, zUp)});
  }
};

class Solver {
  ProblemSetup &ps;
  // Quadrature rule for face/cell integration
  Numeric::QuadRule qr;
  // Differentiation matrix for source terms
  Numeric::DiffMat diffMat;
  // Grid geometry data
  Geometry geom;
  // IO object
  IO::WriterPDI writerPDI;
  // Stencil object for Riemann state reconstruction
  Stencil::Stencil stencil;
  // Boundary condition coefficients
  Physics::BCCoeffs<eqType> bcCoeff;
  // Cell-based data
  ConsDataView U_halo, Uprev_halo, Uhat, Utmp, RHS;
  // Auxiliary data (cell-based but different number of components)
  AuxDataView U_aux;
  // Face-based data
  struct FaceVals {
    FaceDataView KFVM_D_DECL(xDir, yDir, zDir);
    FaceVals(const ProblemSetup &ps_)
        : KFVM_D_DECL(
              xDir("Solver::FaceVals::xDir", KFVM_D_DECL(ps_.nX + 1, ps_.nY, ps_.nZ)),
              yDir("Solver::FaceVals::yDir", KFVM_D_DECL(ps_.nX, ps_.nY + 1, ps_.nZ)),
              zDir("Solver::FaceVals::zDir", KFVM_D_DECL(ps_.nX, ps_.nY, ps_.nZ + 1))) {}
  };
  FaceVals faceVals;
  // Pointwise cell data for source terms
  SourceDataView sourceTerms;
  // Buffers for boundary communication
  BoundaryData bdyData;
  // Struct to manage weno/linear reconstruction
  WenoSelector wenoSelector;
  bool useSparseWeno;
  // Time step information
  int nTS;
  Real time, dt, errEst;
  int plotNum;
  bool writePlot, lastTimeStep;
  // Count RHS evals and step rejections
  int nRhsEval, nRejectUnphys, nRejectThresh;

  // Internal functions
  auto trimCellHalo(ConsDataView v) const {
    return Kokkos::subview(v,
                           KFVM_D_DECL(Kokkos::make_pair(ps.rad, ps.nX + ps.rad),
                                       Kokkos::make_pair(ps.rad, ps.nY + ps.rad),
                                       Kokkos::make_pair(ps.rad, ps.nZ + ps.rad)),
                           Kokkos::ALL);
  }

  auto interiorCellRange() const {
    return Kokkos::MDRangePolicy<ExecSpace, Kokkos::Rank<SPACE_DIM>,
                                 Kokkos::IndexType<idx_t>>(
        {KFVM_D_DECL(0, 0, 0)}, {KFVM_D_DECL(ps.nX, ps.nY, ps.nZ)});
  }

  Real rhsSpeedComm(Real);
  Real glmSpeedComm(Real);
  int timeStepStatusComm(int);
  Real timeStepErrEstComm(Real);

 public:
  Solver(ProblemSetup &);

  void Solve();
  void TakeStep();

  void setIC();

  void evalAuxiliary();
  void evalFlowStats();

  Real evalRHS(ConsDataView, Real);
  void reconRiemStatesHighOrder(ConsDataView);
  void reconRiemStatesSparseWeno(ConsDataView);
  void reconRiemStatesTiledWeno(ConsDataView);
  Real findFluxes();

  void setCellBCs(ConsDataView, Real);
  void setFaceBCs(Real);

  void commCellBCsEW(ConsDataView);
  void commCellBCsNS(ConsDataView);
  void commCellBCsTB(ConsDataView);

  void commFaceBCsEW();
  void commFaceBCsNS();
  void commFaceBCsTB();

  void setWestBCExt(ConsDataView, Real);
  void setEastBCExt(ConsDataView, Real);
  void setSouthBCExt(ConsDataView, Real);
  void setNorthBCExt(ConsDataView, Real);
  void setBottomBCExt(ConsDataView, Real);
  void setTopBCExt(ConsDataView, Real);

  void setWestBCExt(Real);
  void setEastBCExt(Real);
  void setSouthBCExt(Real);
  void setNorthBCExt(Real);
  void setBottomBCExt(Real);
  void setTopBCExt(Real);
};

} // namespace KFVM

#endif
