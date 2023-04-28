#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Geometry.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "numeric/Numeric.H"
#include "stencil/Stencil.H"
#include "physics/Physics_K.H"
#include "io/WriterPDI.H"

namespace KFVM {

  class Solver
  {
    ProblemSetup& ps;
    // Quadrature rule for face/cell integration
    Numeric::QuadRule qr;
    // Grid geometry data
    Geometry geom;
    // IO object
    IO::WriterPDI writerPDI;
    // Stencil object for Riemann state reconstruction
    Stencil::Stencil stencil;
    // Boundary condition coefficients
    Physics::BCCoeffs<eqType> bcCoeff;
    // Field to test whether weno should be activated
    CellDataView wenoSelect;
    // Cell-based data
    ConsDataView U_halo,Uhat,Uprev_halo,Utmp,K;
    // Buffers for boundary communication
//     struct BdyData {
// #if (SPACE_DIM == 3)
//       ConsDataView bottomSrc,nottomDst,topSrc,topDst;
// #endif
//       ConsDataView westSrc,westDst,eastSrc,eastDst;
//       ConsDataView southSrc,southDst,northSrc,northDst;
//       bdyData(const ProblemSetup& ps_):
// #if (SPACE_DIM == 3)
// 	bottomSrc("Solver::BdyData::bottomSrc",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.rad + 1)),
// 	bottomDst("Solver::BdyData::bottomDst",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.rad + 1)),
// 	topSrc("Solver::BdyData::topSrc",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.rad + 1)),
// 	topDst("Solver::BdyData::topDst",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.rad + 1)),
// #endif
// 	westSrc("Solver::BdyData::westSrc",KFVM_D_DECL(ps_.rad + 1,ps_.nY,ps_.nZ)),
// 	westDst("Solver::BdyData::westDst",KFVM_D_DECL(ps_.rad + 1,ps_.nY,ps_.nZ)),
// 	eastSrc("Solver::BdyData::eastSrc",KFVM_D_DECL(ps_.rad + 1,ps_.nY,ps_.nZ)),
// 	eastDst("Solver::BdyData::eastDst",KFVM_D_DECL(ps_.rad + 1,ps_.nY,ps_.nZ)),
// 	southSrc("Solver::BdyData::southSrc",KFVM_D_DECL(ps_.nX,ps_.rad + 1,ps_.nZ)),
// 	southDst("Solver::BdyData::southDst",KFVM_D_DECL(ps_.nX,ps_.rad + 1,ps_.nZ)),
// 	northSrc("Solver::BdyData::northSrc",KFVM_D_DECL(ps_.nX,ps_.rad + 1,ps_.nZ)),
// 	northDst("Solver::BdyData::northDst",KFVM_D_DECL(ps_.nX,ps_.rad + 1,ps_.nZ)),
//       {}
//     };
//     BdyData bdyData;
    // Auxiliary data (cell-based but different number of components)
    AuxDataView U_aux;
    // Face-based data
    struct FaceVals {
      FaceDataView KFVM_D_DECL(xDir,yDir,zDir);
      FaceVals(const ProblemSetup& ps_):
	KFVM_D_DECL(xDir("Solver::FaceVals::xDir",KFVM_D_DECL(ps_.nX + 1,ps_.nY,ps_.nZ)),
		    yDir("Solver::FaceVals::yDir",KFVM_D_DECL(ps_.nX,ps_.nY + 1,ps_.nZ)),
		    zDir("Solver::FaceVals::zDir",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.nZ + 1)))
      {}
    };
    FaceVals faceVals;
    // Time step information
    Real time,dt,errEst;
    bool lastTimeStep;
    // Threshold for WENO reconstruction
    Real wThresh;
    // Count RHS evals and step rejections
    int nRhsEval,nRejectUnphys,nRejectThresh;
    
    // Internal functions
    auto trimCellHalo(ConsDataView v)
    {
      return Kokkos::subview(v,KFVM_D_DECL(Kokkos::make_pair(ps.rad,ps.nX + ps.rad),
					   Kokkos::make_pair(ps.rad,ps.nY + ps.rad),
					   Kokkos::make_pair(ps.rad,ps.nZ + ps.rad)),
			     Kokkos::ALL);
    }
    
    void setIC();
  public:
    Solver(ProblemSetup&);
    void Solve();
    void TakeStep();
    void evalAuxiliary();
    Real evalRHS(ConsDataView,ConsDataView,Real);
    void reconstructRiemannStates(ConsDataView);
    Real findFluxes();
    void setCellBCs(ConsDataView,Real);
    void setWestBCExt(ConsDataView,Real);
    void setEastBCExt(ConsDataView,Real);
    void setSouthBCExt(ConsDataView,Real);
    void setNorthBCExt(ConsDataView,Real);
    void setBottomBCExt(ConsDataView,Real);
    void setTopBCExt(ConsDataView,Real);
    void setFaceBCs(Real);
  };

}

#endif
