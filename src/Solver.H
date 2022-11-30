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
#include "NetCDFWriter.H"
#include "numeric/Numeric.H"
#include "stencil/Stencil.H"
#include "physics/Physics_K.H"

namespace KFVM {

  class Solver
  {
    ProblemSetup& ps;
    // IO object
    NetCDFWriter netCDFWriter;
    // Quadrature rule for face/cell integration
    Numeric::QuadRule qr;
    // Grid geometry data
    Geometry geom;
    // Stencil object for Riemann state reconstruction
    Stencil::Stencil stencil;
    // Boundary condition coefficients
    Physics::BCCoeffs<eqType> bcCoeff;
    // Field to test whether weno should be activated
    CellDataView wenoSelect;
    // Cell-based data
    ConsDataView U_halo,Uhat,Uprev_halo,Utmp,K;
    // Auxiliary data (cell-based but different number of components)
    AuxDataView U_aux;
    // Face-based data
    struct FaceVals {
      FaceDataView KFVM_D_DECL(xDir,yDir,zDir);
      FaceVals(const ProblemSetup& ps_):
	KFVM_D_DECL(xDir("FaceVals::xDir",KFVM_D_DECL(ps_.nX + 1,ps_.nY,ps_.nZ)),
		    yDir("FaceVals::xDir",KFVM_D_DECL(ps_.nX,ps_.nY + 1,ps_.nZ)),
		    zDir("FaceVals::xDir",KFVM_D_DECL(ps_.nX,ps_.nY,ps_.nZ + 1)))
      {}
    };
    FaceVals faceVals;
    // Time step information
    Real time,dt,errEst;
    bool lastTimeStep;
    // Threshold for WENO reconstruction
    Real wThresh;
    // Count RHS evals and step rejections
    int nRhsEval,nRejected;
    
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
    void setFaceBCs(Real);
  };

}

#endif
