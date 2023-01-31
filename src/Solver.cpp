// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "BoundaryConditions_K.H"
#include "NetCDFWriter.H"
#include "numeric/Numeric.H"
#include "numeric/Numeric_K.H"
#include "numeric/RKTypes.H"
#include "physics/EquationTypes.H"
#include "physics/Physics_K.H"
#include "stencil/Stencil_K.H"

#include <Solver.H>

namespace KFVM {
  
  Solver::Solver(ProblemSetup& ps_):
    ps(ps_),
    geom(ps),
    netCDFWriter(ps,geom),
    stencil(ps.gp_lFac),
    wenoSelect("WenoSelector",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U_halo("U",KFVM_D_DECL(ps.nX + 2*ps.rad,
			   ps.nY + 2*ps.rad,
			   ps.nZ + 2*ps.rad)),
    Uhat("Uhat",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    Uprev_halo("Uprev",KFVM_D_DECL(ps.nX + 2*ps.rad,
				   ps.nY + 2*ps.rad,
				   ps.nZ + 2*ps.rad)),
    Utmp("Utmp",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    K("RHS",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U_aux("U_aux",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    faceVals(ps),
    time(ps.initialTime),
    dt(ps.initialDeltaT),
    errEst(1.0),
    wThresh(-1.0),
    lastTimeStep(false),
    nRhsEval(0),
    nRejectUnphys(0),
    nRejectThresh(0)
  {
    setIC();
    evalAuxiliary();
    netCDFWriter.write(U_halo,U_aux,wenoSelect,0,time);

    // FSAL methods need one RHS eval to start
    Real maxVel = evalRHS(U_halo,K,time);
    // Use small CFL just to pick first time step size
    dt = 0.01*ps.cfl*geom.dmin/maxVel;
  }

  // Solve system for full time range
  void Solver::Solve()
  {
    Kokkos::Profiling::pushRegion("Solver::Solve");
    
    // Evolve in time, recording solutions as needed
    // Start at nT=1 since IC is step 0
    for (int nT=1; nT<ps.maxTimeSteps && !lastTimeStep; ++nT) {
      std::printf("Step %d: time = %e\n",nT,time);
      TakeStep();

      // Write out data file if needed
      if (nT%ps.plotFreq == 0 || lastTimeStep || nT == (ps.maxTimeSteps-1) ) {
	evalAuxiliary();
        netCDFWriter.write(U_halo,U_aux,wenoSelect,nT,time);
      }
    }

    std::printf("Time stepping completed\n  %d RHS evals and %d/%d rejected (accuracy/unphysical) steps\n",nRhsEval,nRejectThresh,nRejectUnphys);

    Kokkos::Profiling::popRegion();
  }

  void Solver::evalAuxiliary()
  {
    Kokkos::Profiling::pushRegion("Solver::evalAuxiliary");

    // Range policy only over interior
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,
			    Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});

    auto U = trimCellHalo(U_halo);

    Kokkos::parallel_for("AuxVars",cellRng,Physics::AuxVars<eqType,decltype(U)>(U,U_aux,ps.fluidProp));
    
    Kokkos::Profiling::popRegion();
  }

  void Solver::TakeStep()
  {
    // Pull out RK coefficients
    using RKCoeff = Numeric::RKCoeff<rkType>;
    
    Kokkos::Profiling::pushRegion("Solver::TakeStep");

    // Set range policy for summing registers together
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,
			    Kokkos::IndexType<idx_t>>({KFVM_D_DECL(0,0,0)},
						      {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    // Test if this is the last step, and limit dt as needed
    if (time + dt > ps.finalTime) {
      dt = ps.finalTime - time;
      lastTimeStep = true;
    }

    // Try step with dt, and repeat as needed
    bool accepted = false,firstUnphys = true;
    Real maxVel,v;
    wThresh = ps.fluidProp.wenoThresh*dt;
    for (int nT=0; nT<ps.rejectionLimit; nT++) {
      maxVel = 0.0;
      std::printf("  Attempt %d: dt = %e, wThresh = %e",nT+1,dt,wThresh);

      // Trim halos off
      auto U = trimCellHalo(U_halo);
      auto Uprev = trimCellHalo(Uprev_halo);

      // Need to reset K if this is a repeat of a rejected step
      if (nT > 0) {
	v = evalRHS(Uprev_halo,K,time);
        maxVel = std::fmax(maxVel,v);
      }

      // First stage is special, do it outside loop
      Real betaDt = RKCoeff::beta[0]*dt;
      Real bhatDt = RKCoeff::bhat[0]*dt;
      Kokkos::parallel_for("RKStagePre",cellRng,Numeric::RKFSAL_StagePre_K<decltype(U),decltype(K)>(U,Uhat,Utmp,Uprev,K,betaDt,bhatDt));

      // Loop over stages and update registers
      for (int nS=1; nS<RKCoeff::nStages; nS++) {
	// Extract RK coeffs into local vars to make dispatch happy
	Real delta = RKCoeff::delta[nS];
	Real gam1 = RKCoeff::gam1[nS];
	Real gam2 = RKCoeff::gam2[nS];
	Real gam3 = RKCoeff::gam3[nS];
	Real cDt = RKCoeff::c[nS]*dt;
	bhatDt = RKCoeff::bhat[nS]*dt;
	betaDt = RKCoeff::beta[nS]*dt;

	// Evaluate RHS on U
	v = evalRHS(U_halo,K,time + cDt);
        maxVel = std::fmax(maxVel,v);
      
	// Update registers
	Kokkos::parallel_for("RKStage",cellRng,
			     Numeric::RKFSAL_Stage_K<decltype(U),decltype(K)>(U,Uhat,Utmp,Uprev,K,delta,gam1,gam2,gam3,betaDt,bhatDt));
      }

      // FSAL stage
      v = evalRHS(U_halo,K,time + dt);
      maxVel = std::fmax(maxVel,v);
      bhatDt = RKCoeff::bhatfsal*dt;
      Kokkos::parallel_for("RKStageFSAL",cellRng,Numeric::RKFSAL_StageLast_K<decltype(K)>(Uhat,K,bhatDt));

      // Calculate CFL that ended up being used
      Real cfl = dt*maxVel/geom.dmin;

      // Estimate the error and test positivity
      Real errNew = 0.0,posFlag = 1.0,nDofs = ps.nX*ps.nY*ps.nZ*NUM_VARS;
      Kokkos::parallel_reduce("ErrorEstimate",cellRng,
			      Numeric::RKFSAL_ErrEst_K<decltype(U),decltype(K)>(U,Uhat,ps.atol,ps.rtol),errNew,Kokkos::Min<Real>(posFlag));
      errNew = 1.0/std::sqrt(errNew/nDofs);

      // Set a new time step size
      Real dtfac = std::pow(errNew,RKCoeff::ep1)*std::pow(errEst,RKCoeff::ep2);
      dtfac = 1.0 + std::atan(dtfac - 1.0);
      if (posFlag < 0.0) {
	// Solution is unphysical, reject and reduce dt
        std::printf(", cfl = %f\n    Rejected: Unphysical\n",cfl);
	// first set to quarter of max cfl, then start halving
	dt = firstUnphys ? 0.25*ps.cfl*geom.dmin/maxVel : dt/2.0;
	firstUnphys = false;
	// Force full weno usage on failure
        wThresh = -std::numeric_limits<Real>::max();
	nRejectUnphys++;
      } else if (dtfac >= ps.rejectionThresh) {
	// Step is accepted
        std::printf(", cfl = %f\n",cfl);
	accepted = true;
	time += dt;
	Real dterr = dt*dtfac;
        Real dtcfl = ps.cfl*geom.dmin/maxVel;
        dt = std::fmin(dterr,dtcfl); // Limit to max cfl
	errEst = errNew;

	// Update the weno selector
	Kokkos::parallel_for("WenoSelector",cellRng,Numeric::RK_WenoSelect_K<decltype(U),decltype(wenoSelect)>(U,Uprev,wenoSelect));
	
	Kokkos::deep_copy(Uprev,U);
	break;
      } else {
	// otherwise step is rejected, try again with new smaller dt
        std::printf(", cfl = %f\n    Rejected: Tolerance\n",cfl);
	dt *= dtfac;
	nRejectThresh++;
      }
    }

    // Error out if step was rejected too many times or has stagnated
    if (!accepted) {
      std::printf("Warning: Time step was rejected too many times\n");
      lastTimeStep = true;
    }
    if (dt < ps.initialDeltaT) {
      std::printf("Warning: Time step stagnated\n");
      lastTimeStep = true;
    }

    Kokkos::Profiling::popRegion();
  }
  
  Real Solver::evalRHS(ConsDataView sol_halo,ConsDataView rhs,Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::evalRHS");

    nRhsEval++;
    
    // Set BCs on cell averages
    setCellBCs(sol_halo,t);
    
    // Reconstruct face states
    reconstructRiemannStates(sol_halo);

    // Calculate cleaning speed for GLM method
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    if (eqType == EquationType::MHD_GLM) {
      ps.fluidProp.ch_glm = 0.0;
      Kokkos::parallel_reduce("CalculateCH_GLM",cellRng,
			      Physics::SpeedEstimate_K<eqType>(KFVM_D_DECL(faceVals.xDir,faceVals.yDir,faceVals.zDir),ps.fluidProp),
			      Kokkos::Max<Real>(ps.fluidProp.ch_glm));
    }
    
    // Set BCs on Riemann states
    setFaceBCs(t);
    
    // Call Riemann solver
    Real vEW = 0.0,vNS = 0.0,vTB = 0.0;

    // East/West faces
    auto fluxRng_EW = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX + 1,ps.nY,ps.nZ)});
    Kokkos::parallel_reduce("RiemannSolver::EW",fluxRng_EW,
			    Physics::RiemannSolverX_K<eqType,rsType>(faceVals.xDir,ps.fluidProp),
			    Kokkos::Max<Real>(vEW));

    // North/South faces
    auto fluxRng_NS = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY + 1,ps.nZ)});
    Kokkos::parallel_reduce("RiemannSolver::NS",fluxRng_NS,
			    Physics::RiemannSolverY_K<eqType,rsType>(faceVals.yDir,ps.fluidProp),
			    Kokkos::Max<Real>(vNS));
    
#if (SPACE_DIM == 3)
    // Top/Bottom faces
    auto fluxRng_TB = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({0,0,0},{ps.nX,ps.nY,ps.nZ + 1});
    Kokkos::parallel_reduce("RiemannSolver::TB",fluxRng_TB,
			    Physics::RiemannSolverZ_K<eqType,rsType>(faceVals.zDir,ps.fluidProp),
			    Kokkos::Max<Real>(vTB));
#endif

    // Reduce max velocities from each direction
    Real maxVel = std::fmax(vEW,std::fmax(vNS,vTB));

    // Integrate fluxes and store into rhs
    Kokkos::parallel_for("IntegrateFlux",cellRng,
			 Numeric::IntegrateFlux_K<decltype(rhs)>
			 (rhs,
			  KFVM_D_DECL(faceVals.xDir,faceVals.yDir,faceVals.zDir),
			  qr.ab,qr.wt,geom));
    
    // Fill in source terms
    auto sol = trimCellHalo(sol_halo);
    Kokkos::parallel_for("SourceTerms",cellRng,
			 Physics::SourceTerms_K<eqType,decltype(sol),decltype(rhs)>(sol,rhs,ps.fluidProp,geom,t));
    
    Kokkos::Profiling::popRegion();
    return maxVel;
  }

  void Solver::reconstructRiemannStates(ConsDataView sol_halo)
  {
    // Subviews of cell data and Riemann states to simplify indexing
    auto U = trimCellHalo(sol_halo);

    // Allocate views for stencil values and smoothness indicators
    Stencil::WorkView stenWork("StenWork",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ));
    
    auto cellRng =
      Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});

    // Weno reconstruction
    Kokkos::parallel_for("FaceRecon",cellRng,
			 Stencil::KernelWenoRecon_K<decltype(U),decltype(wenoSelect)>
			 (U,wenoSelect,wThresh,
			  KFVM_D_DECL(faceVals.xDir,
				      faceVals.yDir,
				      faceVals.zDir),
			  stenWork,
			  KFVM_D_DECL(stencil.lOff,
				      stencil.tOff,
				      stencil.ttOff),
			  stencil.subIdx,
			  stencil.faceWeights,
			  stencil.derivWeights,
                          ps.fluidProp));

    // Enforce positivity of Riemann states
    Kokkos::parallel_for("PosPres",cellRng,
        		 Physics::PositivityPreserve_K<eqType,decltype(U)>
        		 (U,
        		  KFVM_D_DECL(faceVals.xDir,
        			      faceVals.yDir,
        			      faceVals.zDir),
        		  ps.fluidProp));
  }

void Solver::setCellBCs(ConsDataView sol_halo,Real t)
  {
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
			   CellBcWest_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.x)>
			   (sol_halo,bcCoeff.x,ps.rad,ps.nX));
      break;
    case BCType::periodic:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BCType::periodic>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::West",bndRng_EW,
			   CellBcWest_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nX,t));
      break;
    default:
      std::printf("Warning: Western cell BC undefined. How did this even compile?\n");
    }

    // Eastern Boundary
    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.x)>
			   (sol_halo,bcCoeff.x,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::East",bndRng_EW,
			   CellBcEast_K<decltype(sol_halo),BCType::user>
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
			   CellBcSouth_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.y)>
			   (sol_halo,bcCoeff.y,ps.rad,ps.nY));
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
			   CellBcNorth_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.y)>
			   (sol_halo,bcCoeff.y,ps.rad,ps.nY));
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
			   CellBcBottom_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.z)>
			   (sol_halo,bcCoeff.z,ps.rad,ps.nZ));
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
			   CellBcTop_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.z)>
			   (sol_halo,bcCoeff.z,ps.rad,ps.nZ));
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
			   FaceBcWest_K<decltype(westBnd),BCType::outflow>(westBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::West::Reflecting",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),BCType::reflecting,decltype(bcCoeff.x)>(westBnd,bcCoeff.x));
      break;
    case BCType::periodic:
      Kokkos::parallel_for("FaceBCs::West::Periodic",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),BCType::periodic>(westBnd,eastBnd));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::West::User",bndRng_EW,
			   FaceBcWest_K<decltype(westBnd),BCType::user>(westBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Western face BC undefined. How did this even compile?\n");
    }

    // Eastern Boundary
    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::East::Outflow",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),BCType::outflow>(eastBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::East::Reflecting",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),BCType::reflecting,decltype(bcCoeff.x)>(eastBnd,bcCoeff.x));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::East::User",bndRng_EW,
			   FaceBcEast_K<decltype(eastBnd),BCType::user>(eastBnd,geom,qr.ab,t));
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
			   FaceBcSouth_K<decltype(southBnd),BCType::periodic>(southBnd,northBnd));
      break;
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::South::Outflow",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),BCType::outflow>(southBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::South::Reflecting",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),BCType::reflecting,decltype(bcCoeff.y)>(southBnd,bcCoeff.y));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::South::User",bndRng_NS,
			   FaceBcSouth_K<decltype(southBnd),BCType::user>(southBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Southern? face BC undefined. How did this even compile?\n");
    }

    // Northern Boundary
    switch (ps.bcType[FaceLabel::north]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::North::Outflow",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),BCType::outflow>(northBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::North::Reflecting",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),BCType::reflecting,decltype(bcCoeff.y)>(northBnd,bcCoeff.y));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::North::User",bndRng_NS,
			   FaceBcNorth_K<decltype(northBnd),BCType::user>(northBnd,geom,qr.ab,t));
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
			   FaceBcBottom_K<decltype(bottomBnd),BCType::periodic>(bottomBnd,topBnd));
      break;
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Bottom::Outflow",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::outflow>(bottomBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Bottom::Reflecting",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::reflecting,decltype(bcCoeff.z)>(bottomBnd,bcCoeff.z));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Bottom::User",bndRng_TB,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::user>(bottomBnd,geom,qr.ab,t));
      break;
    default:
      std::printf("Warning: Bottom? face BC undefined. How did this even compile?\n");
    }

    // Top Boundary
    switch (ps.bcType[FaceLabel::top]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Top::Outflow",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),BCType::outflow>(topBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Top::Reflecting",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),BCType::reflecting,decltype(bcCoeff.z)>(topBnd,bcCoeff.z));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Top::User",bndRng_TB,
			   FaceBcTop_K<decltype(topBnd),BCType::user>(topBnd,geom,qr.ab,t));
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

    // Fill U using user specified initial condition
    auto U = trimCellHalo(U_halo);
    auto cellRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_for("IntegrateIC",cellRng,Numeric::IntegrateIC_K<decltype(U)>
			 (U,qr.ab,qr.wt,geom));

    // Set Uprev to also be IC
    // Needed for error control in timestepper
    Kokkos::deep_copy(Uprev_halo,U_halo);

    // First step always uses WENO everywhere
    // -> set wenoSelect to large number everywhere
    Kokkos::deep_copy(wenoSelect,100.0);

    Kokkos::Profiling::popRegion();
  }
  
} // end namespace KFVM
