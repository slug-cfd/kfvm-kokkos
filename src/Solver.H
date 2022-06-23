#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include <Kokkos_View.hpp>

#include "Dimension.H"
#include "Geometry.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "Stencil.H"
#include "NetCDFWriter.H"

namespace KFVM {

  class Solver 
  {
    const ProblemSetup& ps;
    // IO object
    NetCDFWriter netCDFWriter;
    // Grid geometry data
    Geometry geom;
    // Cell-based data
    CellDataView U_halo,U1_halo,U2_halo,U3_halo,U4_halo,K,Ktil;
    // Face-based data
    FaceDataView FaceVals;
    // Time step information
    Real time,dt;
    bool lastTimeStep;
    // Internal functions
    auto trimCellHalo(CellDataView v)
    {
      return Kokkos::subview(v,KFVM_D_DECL(
					   Kokkos::make_pair(int(ps.rad),int(ps.nX + ps.rad)),
					   Kokkos::make_pair(int(ps.rad),int(ps.nY + ps.rad)),
					   Kokkos::make_pair(int(ps.rad),int(ps.nZ + ps.rad))),
			     Kokkos::ALL);
    }
    auto trimFaceHalo(FaceDataView v)
    {
#if (SPACE_DIM == 2)
      return Kokkos::subview(v,
			     Kokkos::make_pair(1,int(ps.nX + 1)),
			     Kokkos::make_pair(1,int(ps.nY + 1)),
			     Kokkos::ALL,
			     Kokkos::ALL,
			     Kokkos::ALL);
#else
      return Kokkos::subview(v,
			     Kokkos::make_pair(1,int(ps.nX + 1)),
			     Kokkos::make_pair(1,int(ps.nY + 1)),
			     Kokkos::make_pair(1,int(ps.nZ + 1)),
			     Kokkos::ALL,
			     Kokkos::ALL,
			     Kokkos::ALL,
			     Kokkos::ALL);
#endif
    }
    void setIC();
    Real evalRHS(CellDataView,CellDataView,Real);
    Real findFluxes();
    void setCellBCs(CellDataView,Real);
    void setFaceBCs(Real);
  public:
    Solver(const ProblemSetup&);
    void TakeStep();
    void Solve();
  };

}

#endif
