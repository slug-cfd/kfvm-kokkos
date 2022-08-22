#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include <Kokkos_View.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Geometry.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "NetCDFWriter.H"
#include "numeric/Numeric.H"
#include "stencil/Stencil.H"

namespace KFVM {

  class Solver 
  {
    const ProblemSetup& ps;
    // IO object
    NetCDFWriter netCDFWriter;
    // Quadrature rule for face/cell integration
    Numeric::QuadRule qr;
    // Grid geometry data
    Geometry geom;
    // Stencil object for Riemann state reconstruction
    Stencil::Stencil stencil;
    // Cell-based data
    CellDataView U_halo,U1_halo,U2_halo,U3_halo,U4_halo,K,Ktil;
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
    Real time,dt;
    bool lastTimeStep;
    // Internal functions
    auto trimCellHalo(CellDataView v)
    {
      return Kokkos::subview(v,KFVM_D_DECL(Kokkos::make_pair(ps.rad,ps.nX + ps.rad),
					   Kokkos::make_pair(ps.rad,ps.nY + ps.rad),
					   Kokkos::make_pair(ps.rad,ps.nZ + ps.rad)),
			     Kokkos::ALL);
    }
    
    void setIC();
  public:
    Solver(const ProblemSetup&);
    void TakeStep();
    void Solve();
    Real evalRHS(CellDataView,CellDataView,Real);
    void reconstructRiemannStates(CellDataView);
    Real findFluxes();
    void setCellBCs(CellDataView,Real);
    void setFaceBCs(Real);
  };

}

#endif
