// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

#include <Kokkos_Array.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_Core.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Macros.hpp>
#include <Kokkos_Rank.hpp>
#include <Kokkos_View.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <impl/Kokkos_HostThreadTeam.hpp>
#include <impl/Kokkos_Profiling.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "BoundaryConditions_K.H"
#include "NetCDFWriter.H"
#include "numeric/Numeric.H"
#include "numeric/Numeric_K.H"
#include "hydro/Hydro_K.H"
#include "stencil/Stencil_K.H"

#include <Solver.H>

namespace KFVM {
  
  Solver::Solver(const ProblemSetup& ps_):
    ps(ps_),
    netCDFWriter(ps),
    geom(ps),
    stencil(ps.gp_lFac),
    U_halo("U",        KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    U1_halo("U_stage1",KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    U2_halo("U_stage2",KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    U3_halo("U_stage3",KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    U4_halo("U_stage4",KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    K("RHS",       KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    Ktil("RHS_til",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    faceVals(ps),
    time(ps.initialTime),
    dt(ps.initialDeltaT),
    lastTimeStep(false)
  {
    setIC();
    netCDFWriter.write(U_halo,0,time);
  }

  // Solve system for full time range
  void Solver::Solve()
  {
    Kokkos::Profiling::pushRegion("Solver::Solve");
    
    // Evolve in time, recording solutions as needed
    // Start at nT=1 since IC is step 0
    for (int nT=1; nT<ps.maxTimeSteps && !lastTimeStep; ++nT) {
      std::printf("Step %d: ",nT);
      TakeStep();
      if (nT%ps.plotFreq == 0 || lastTimeStep || nT == (ps.maxTimeSteps-1) ) {
        netCDFWriter.write(U_halo,nT,time);
      }
    }

    Kokkos::Profiling::popRegion();
  }

  void Solver::TakeStep()
  {
    Kokkos::Profiling::pushRegion("Solver::TakeStep");

    // Set range policy for summing stages together
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    
    // Evaluate RHS and set dt
    Real maxVel = evalRHS(U_halo,K,time);
    dt = std::fmin(ps.cfl*std::fmin(ps.dx,ps.dy)/maxVel,2.0*dt);
    if (time + dt > ps.finalTime) {
      dt = ps.finalTime - time;
      lastTimeStep = true;
    } else if (dt < ps.initialDeltaT/10.0) {
      std::printf("Warning: Time step has stagnated\n");
      lastTimeStep = true;
    }
    std::printf("time = %e, dt = %e\n",time,dt);

    // First stage
    auto U = trimCellHalo(U_halo);
    auto U1 = trimCellHalo(U1_halo);
    Kokkos::parallel_for("RKStage_1",cellRng,
			 Numeric::SSP45_S1_K<decltype(U),decltype(K),5>
			 (U1,U,K,Numeric::SSPCoeff<5>(),dt));

    // Second stage
    evalRHS(U1_halo,K,time + dt/2.0);
    auto U2 = trimCellHalo(U2_halo);
    Kokkos::parallel_for("RKStage_2",cellRng,
			 Numeric::SSP45_S2_K<decltype(U),decltype(K),5>
			 (U2,U,U1,K,Numeric::SSPCoeff<5>(),dt));

    // Third stage
    evalRHS(U2_halo,K,time + dt/2.0);
    auto U3 = trimCellHalo(U3_halo);
    Kokkos::parallel_for("RKStage_3",cellRng,
			 Numeric::SSP45_S3_K<decltype(U),decltype(K),5>
			 (U3,U,U2,K,Numeric::SSPCoeff<5>(),dt));

    // Fourth stage
    evalRHS(U3_halo,K,time + dt/2.0);
    auto U4 = trimCellHalo(U4_halo);
    Kokkos::parallel_for("RKStage_4",cellRng,
			 Numeric::SSP45_S4_K<decltype(U),decltype(K),5>
			 (U4,U,U3,K,Numeric::SSPCoeff<5>(),dt));

    // Fifth stage
    evalRHS(U4_halo,Ktil,time + dt/2.0);
    Kokkos::parallel_for("RKStage_5",cellRng,
			 Numeric::SSP45_S5_K<decltype(U),decltype(K),5>
			 (U,U2,U3,U4,K,Ktil,Numeric::SSPCoeff<5>(),dt));

    time += dt;

    Kokkos::Profiling::popRegion();
  }

  Real Solver::evalRHS(CellDataView sol_halo,CellDataView rhs,Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::evalRHS");
    
    // Set BCs on cell averages
    setCellBCs(sol_halo,t);

    // Reconstruct face states
    reconstructRiemannStates(sol_halo);
    
    // Set BCs on Riemann states
    setFaceBCs(t);
    
    // Call Riemann solver
    Real vEW = 0.0,vNS = 0.0,vTB = 0.0;

    // East/West faces
    auto fluxRng_EW = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX + 1,ps.nY,ps.nZ)});
    Kokkos::parallel_reduce("RiemannSolver::EW",fluxRng_EW,
			    Hydro::RiemannSolverX_K(faceVals.xDir,1.4),
			    Kokkos::Max<Real>(vEW));

    // North/South faces
    auto fluxRng_NS = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY + 1,ps.nZ)});
    Kokkos::parallel_reduce("RiemannSolver::NS",fluxRng_NS,
			    Hydro::RiemannSolverY_K(faceVals.yDir,1.4),
			    Kokkos::Max<Real>(vNS));
    
#if (SPACE_DIM == 3)
    // Top/Bottom faces
    auto fluxRng_TB = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({0,0,0},{ps.nX,ps.nY,ps.nZ + 1});
    Kokkos::parallel_reduce("RiemannSolver::TB",fluxRng_TB,
			    Hydro::RiemannSolverZ_K(faceVals.zDir,1.4),
			    Kokkos::Max<Real>(vTB));
#endif

    // Reduce max velocities from each direction
    Real maxVel = std::fmax(vEW,std::fmax(vNS,vTB));

    // Integrate fluxes and store into rhs
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_for("IntegrateFlux",cellRng,
			 Numeric::IntegrateFlux_K<decltype(rhs)>
			 (rhs,
			  KFVM_D_DECL(faceVals.xDir,faceVals.yDir,faceVals.zDir),
			  qr.ab,qr.wt,geom));
    
    Kokkos::Profiling::popRegion();
    return maxVel;
  }

  void Solver::reconstructRiemannStates(CellDataView sol_halo)
  {
    // Subviews of cell data and Riemann states to simplify indexing
    auto U = trimCellHalo(sol_halo);

    // Allocate views for stencil values and smoothness indicators
    Stencil::WorkView stenWork("StenWork",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ));
    
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});

    // Weno reconstruction
    // Kokkos::parallel_for("FaceRecon",cellRng,
    // 			 Stencil::KernelWenoRecon_K<decltype(U)>
    // 			 (U,
    // 			  KFVM_D_DECL(faceVals.xDir,
    // 				      faceVals.yDir,
    // 				      faceVals.zDir),
    // 			  stenWork,
    // 			  KFVM_D_DECL(stencil.lOff,
    // 				      stencil.tOff,
    // 				      stencil.ttOff),
    // 			  stencil.subIdx,
    // 			  stencil.faceWeights,
    // 			  stencil.derivWeights));

    // Linear Kernel reconstruction
    Kokkos::parallel_for("FaceRecon",cellRng,
			 Stencil::KernelLinearRecon_K<decltype(U)>
			 (U,KFVM_D_DECL(faceVals.xDir,
					faceVals.yDir,
					faceVals.zDir),
			  stenWork,
			  KFVM_D_DECL(stencil.lOff,
				      stencil.tOff,
				      stencil.ttOff),
			  stencil.faceWeights));

    // MinMod reconstruction
    // Kokkos::parallel_for("FaceRecon",cellRng,
    // 			 Stencil::MinModRecon_K<decltype(U)>
    // 			 (U,KFVM_D_DECL(faceVals.xDir,
    // 					faceVals.yDir,
    // 					faceVals.zDir)));

    // Enforce positivity of Riemann states
    Kokkos::parallel_for("PosPres",cellRng,
			 Hydro::PositivityPreserve_K<decltype(U)>
			 (U,
			  KFVM_D_DECL(faceVals.xDir,
				      faceVals.yDir,
				      faceVals.zDir),
			  1.4));
  }

void Solver::setCellBCs(CellDataView sol_halo,Real t)
  {
    using BoundaryConditions::BCType;
    using BoundaryConditions::CellBcWest_K;
    using BoundaryConditions::CellBcEast_K;
    using BoundaryConditions::CellBcSouth_K;
    using BoundaryConditions::CellBcNorth_K;
    using BoundaryConditions::CellBcBottom_K;
    using BoundaryConditions::CellBcTop_K;
    
    Kokkos::Profiling::pushRegion("Solver::setCellBCs");

#if (SPACE_DIM == 2)
    auto bndRng_EW = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad + ps.nY});
    auto bndRng_NS = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({0,ps.nX + 2*ps.rad});
#else
    auto bndRng_EW = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad},{ps.nY + ps.rad,ps.nZ + ps.rad});
    auto bndRng_NS = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,ps.rad},{ps.nX + 2*ps.rad,ps.nZ + ps.rad});
    auto bndRng_TB = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX + 2*ps.rad,ps.nY + 2*ps.rad});
#endif

    // Western Boundary
    switch (ps.bcType[FaceLabel::west]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BoundaryConditions::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BoundaryConditions::reflecting>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::periodic:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BoundaryConditions::periodic>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BoundaryConditions::user>
			   (sol_halo,geom,ps.rad,ps.nX,t));
      break;
    default:
      std::printf("Warning: Western cell BC undefined. How did this even compile?\n");
    }

    // Eastern Boundary
    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BoundaryConditions::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BoundaryConditions::reflecting>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BoundaryConditions::user>
			   (sol_halo,geom,ps.rad,ps.nX,t));
      break;
    default:
      if (ps.bcType[FaceLabel::east] != BCType::periodic) {
	std::printf("Warning: Eastern cell BC undefined. Is west set to periodic and east not?\n");
      }
    }

    // Southern Boundary
    switch (ps.bcType[FaceLabel::south]) {
    case BCType::periodic :
      Kokkos::parallel_for("CellBCs::South",bndRng_NS,
			   CellBcSouth_K<decltype(sol_halo),BCType::periodic>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::outflow :
      Kokkos::parallel_for("CellBCs::South",bndRng_NS,
			   CellBcSouth_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::reflecting :
      Kokkos::parallel_for("CellBCs::South",bndRng_NS,
			   CellBcSouth_K<decltype(sol_halo),BCType::reflecting>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::user :
      Kokkos::parallel_for("CellBCs::South",bndRng_NS,
			   CellBcSouth_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nY,t));
      break;
    default:
      std::printf("Warning: Southern cell BC undefined. How did this even compile?\n");
    }

    // Northern Boundary
    switch (ps.bcType[FaceLabel::north]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::North",bndRng_NS,
			   CellBcNorth_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::North",bndRng_NS,
			   CellBcNorth_K<decltype(sol_halo),BCType::reflecting>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::North",bndRng_NS,
			   CellBcNorth_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nY,t));
      break;
    default:
      if (ps.bcType[FaceLabel::north] != BCType::periodic) {
	std::printf("Warning: Northern cell BC undefined. Is south set to periodic and north not?\n");
      }
    }
    
#if (SPACE_DIM == 3)
    // Bottom Boundary
    switch (ps.bcType[FaceLabel::bottom]) {
    case BCType::periodic :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng_TB,
			   CellBcBottom_K<decltype(sol_halo),BCType::periodic>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::outflow :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng_TB,
			   CellBcBottom_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::reflecting :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng_TB,
			   CellBcBottom_K<decltype(sol_halo),BCType::reflecting>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::user :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng_TB,
			   CellBcBottom_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nZ,t));
      break;
    default:
      std::printf("Warning: Bottom? cell BC undefined. How did this even compile?\n");
    }

    // Top Boundary
    switch (ps.bcType[FaceLabel::top]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::Top",bndRng_TB,
			   CellBcTop_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::Top",bndRng_TB,
			   CellBcTop_K<decltype(sol_halo),BCType::reflecting>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::Top",bndRng_TB,
			   CellBcTop_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nZ,t));
      break;
    default:
      if (ps.bcType[FaceLabel::top] != BCType::periodic) {
	std::printf("Warning: Top cell BC undefined. Is bottom set to periodic and top not?\n");
      }
    }
#endif

    Kokkos::Profiling::popRegion();
  }

  void Solver::setFaceBCs(Real t)
  {
    using BoundaryConditions::BCType;
    using BoundaryConditions::FaceBcWest_K;
    using BoundaryConditions::FaceBcEast_K;
    using BoundaryConditions::FaceBcSouth_K;
    using BoundaryConditions::FaceBcNorth_K;
    using BoundaryConditions::FaceBcBottom_K;
    using BoundaryConditions::FaceBcTop_K;
    
    (void) t;
    Kokkos::Profiling::pushRegion("Solver::setFaceBCs");

    // Set ranges for each face
    // need to specialize on dimension since MDRangepolicy will not
    // decay to 1D
#if (SPACE_DIM == 2)
    auto bndRng_EW = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nY});
    auto bndRng_NS = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nX});
#else
    auto bndRng_EW = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nY,ps.nZ});
    auto bndRng_NS = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nZ});
    auto bndRng_TB = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nY});
#endif

    // Extract subviews of each boundary
    auto westBnd = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto eastBnd = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto southBnd = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto northBnd = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
#if (SPACE_DIM == 3)
    auto bottomBnd = Kokkos::subview(faceVals.zDir,
				     Kokkos::ALL,Kokkos::ALL,0,
				     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    auto topBnd = Kokkos::subview(faceVals.zDir,
				  Kokkos::ALL,Kokkos::ALL,ps.nZ,
				  Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
#endif
    
    // Western Boundary
    switch (ps.bcType[FaceLabel::west]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::West::Outflow",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),
			   BCType::outflow>(westBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::West::Reflecting",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),
			   BCType::reflecting>(westBnd));
      break;
    case BCType::periodic:
      Kokkos::parallel_for("FaceBCs::West::Periodic",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),
			   BCType::periodic>(westBnd,eastBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::West::User",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),
			   BCType::user>(westBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Western face BC undefined. How did this even compile?\n");
    }

    // Eastern Boundary
    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::East::Outflow",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),
			   BCType::outflow>(eastBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::East::Reflecting",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),
			   BCType::reflecting>(eastBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::East::User",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),
			   BCType::user>(eastBnd,geom,qr.ab,t));
      break;
    default:
      if (ps.bcType[FaceLabel::east] != BCType::periodic) {
	std::printf("Warning: Eastern face BC undefined.\n");
      }
    }

    // Southern Boundary
    switch (ps.bcType[FaceLabel::south]) {
    case BCType::periodic:
      Kokkos::parallel_for("FaceBCs::South::Periodic",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),
			   BCType::periodic>(southBnd,northBnd));
      break;
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::South::Outflow",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),
			   BCType::outflow>(southBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::South::Reflecting",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),
			   BCType::reflecting>(southBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::South::User",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),
			   BCType::user>(southBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Southern? face BC undefined. How did this even compile?\n");
    }

    // Northern Boundary
    switch (ps.bcType[FaceLabel::north]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::North::Outflow",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),
			   BCType::outflow>(northBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::North::Reflecting",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),
			   BCType::reflecting>(northBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::North::Outflow",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),
			   BCType::outflow>(northBnd));
      // Kokkos::parallel_for("FaceBCs::North::User",bndRng_NS,
      // 			   FaceBcNorth_K<decltype(northBnd),
      // 			   BCType::user>(northBnd,geom,qr.ab,t));
      break;
    default:
      if (ps.bcType[FaceLabel::north] != BCType::periodic) {
	std::printf("Warning: Northern face BC undefined.\n");
      }
    }
    
#if (SPACE_DIM == 3)
    // Bottom Boundary
    switch (ps.bcType[FaceLabel::bottom]) {
    case BCType::periodic:
      Kokkos::parallel_for("FaceBCs::Bottom::Periodic",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),
			   BCType::periodic>(bottomBnd,topBnd));
      break;
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Bottom::Outflow",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),
			   BCType::outflow>(bottomBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Bottom::Reflecting",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),
			   BCType::reflecting>(bottomBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Bottom::User",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),
			   BCType::user>(bottomBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Bottom? face BC undefined. How did this even compile?\n");
    }

    // Top Boundary
    switch (ps.bcType[FaceLabel::top]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Top::Outflow",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),
			   BCType::outflow>(topBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Top::Reflecting",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),
			   BCType::reflecting>(topBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Top::User",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),
			   BCType::user>(topBnd,geom,qr.ab,t));
      break;
    default:
      if (ps.bcType[FaceLabel::top] != BCType::periodic) {
	std::printf("Warning: Top face BC undefined.\n");
      }
    }
#endif

    Kokkos::Profiling::popRegion();
  }

  void Solver::setIC()
  {
    Kokkos::Profiling::pushRegion("Solver::setIC");
    
    auto U = trimCellHalo(U_halo);
    auto cellRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_for("IntegrateIC",cellRng,Numeric::IntegrateIC_K<decltype(U)>
			 (U,qr.ab,qr.wt,geom));

    Kokkos::Profiling::popRegion();
  }
  
} // end namespace KFVM
