#ifndef SOLVER_H_
#define SOLVER_H_

// File: Solver.H
// Purpose: The solver class holds and manages all internal data,
//          and is responsible for evolving the simulation through
//          time.

#include "Dimension.H"
#include "Geometry.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "Stencil.H"

namespace KFVM {

  class Solver 
  {
    const ProblemSetup& ps;
    // Grid geometry data
    Geometry geom;
    // Cell-based data
    CellDataView U,U1,U2,U3,U4,K,Ktil;
    // Face-based data
    FaceDataView RStateX,RStateY,FluxX,FluxY;
    // Time step information
    Real time,deltaT;
    bool lastTimeStep;
    // Internal functions
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
