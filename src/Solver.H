#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include <Kokkos_View.hpp>


#include <Definitions.H>
#include <Dimension.H>
#include <Geometry.H>
#include <SimVar.H>
#include <Types.H>
#include <ProblemSetup.H>
#include <NetCDFWriter.H>
#include <Numeric.H>
#include <Stencil.H>

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
    FaceDataView FaceVals;
    // Cell data expanded for reconstruction stencils
    Stencil::StenValsView StenVals;
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
    auto trimFaceHalo(FaceDataView v)
    {
#if (SPACE_DIM == 2)
      return Kokkos::subview(v,
			     Kokkos::make_pair(1,ps.nX + 1),
			     Kokkos::make_pair(1,ps.nY + 1),
			     Kokkos::ALL,
			     Kokkos::ALL,
			     Kokkos::ALL);
#else
      return Kokkos::subview(v,
			     Kokkos::make_pair(1,ps.nX + 1),
			     Kokkos::make_pair(1,ps.nY + 1),
			     Kokkos::make_pair(1,ps.nZ + 1),
			     Kokkos::ALL,
			     Kokkos::ALL,
			     Kokkos::ALL);
#endif
    }
    
    void setIC();
  public:
    Solver(const ProblemSetup&);
    void TakeStep();
    void Solve();
    Real evalRHS(CellDataView,CellDataView,Real);
    Real findFluxes();
    void setCellBCs(CellDataView,Real);
    void setFaceBCs(Real);
  };

}

#endif
