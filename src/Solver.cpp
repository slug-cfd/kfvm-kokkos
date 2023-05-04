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
#include "FluidProperties.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "PrinterMPI.H"
#include "Geometry.H"
#include "BoundaryConditions_K.H"
#include "BoundaryData.H"
#include "numeric/Numeric.H"
#include "numeric/Numeric_K.H"
#include "numeric/RKTypes.H"
#include "physics/EquationTypes.H"
#include "physics/Physics_K.H"
#include "stencil/Stencil_K.H"
#include "io/WriterPDI.H"

#include <Solver.H>
#include <mpi.h>

#define KFVM_MPI_TAG 0

namespace KFVM {
  
  Solver::Solver(ProblemSetup& ps_):
    ps(ps_),
    geom(ps),
    writerPDI(ps,geom),
    stencil(ps.gp_lFac),
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
    bdyData(ps),
    wenoSelector(ps),
    useSparseWeno(false),
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
    writerPDI.write(U_halo,U_aux,wenoSelector.wenoFlagView,0,time);

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
      PrintSingle(ps,"Step %d: time = %e\n",nT,time);
      TakeStep();

      // Write out data file if needed
      if (nT%ps.plotFreq == 0 || lastTimeStep || nT == (ps.maxTimeSteps-1) ) {
	evalAuxiliary();
        writerPDI.write(U_halo,U_aux,wenoSelector.wenoFlagView,nT,time);
      }
    }

    PrintSingle(ps,"Time stepping completed\n  %d RHS evals and %d/%d rejected (accuracy/unphysical) steps\n",nRhsEval,nRejectThresh,nRejectUnphys);

    Kokkos::Profiling::popRegion();
  }

  void Solver::evalAuxiliary()
  {
    Kokkos::Profiling::pushRegion("Solver::evalAuxiliary");

    auto U = trimCellHalo(U_halo);
    auto cellRng = interiorCellRange();

    Kokkos::parallel_for("Solver::evalAuxiliary",cellRng,
			 Physics::AuxVars<eqType,decltype(U)>
			 (U,U_aux,ps.fluidProp));
    
    Kokkos::Profiling::popRegion();
  }

  void Solver::TakeStep()
  {
    // Pull out RK coefficients
    using RKCoeff = Numeric::RKCoeff<rkType>;
    
    Kokkos::Profiling::pushRegion("Solver::TakeStep");

    // Set range policy for summing registers together
    auto cellRng = interiorCellRange();
    
    // Test if this is the last step, and limit dt as needed
    if (time + dt > ps.finalTime) {
      dt = ps.finalTime - time;
      lastTimeStep = true;
    }

    // Try step with dt, and repeat as needed
    bool accepted = false,firstUnphys = true;
    Real maxVelLoc,v;
    wThresh = ps.fluidProp.wenoThresh*dt;
    for (int nT=0; nT<ps.rejectionLimit; nT++) {
      maxVelLoc = 0.0;
      PrintSingle(ps,"  Attempt %d: dt = %e, wThresh = %e",nT+1,dt,wThresh);

      // Trim halos off
      auto U = trimCellHalo(U_halo);
      auto Uprev = trimCellHalo(Uprev_halo);

      // Need to reset K if this is a repeat of a rejected step
      if (nT > 0) {
	v = evalRHS(Uprev_halo,K,time);
        maxVelLoc = std::fmax(maxVelLoc,v);
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
        maxVelLoc = std::fmax(maxVelLoc,v);
      
	// Update registers
	Kokkos::parallel_for("RKStage",cellRng,
			     Numeric::RKFSAL_Stage_K<decltype(U),decltype(K)>(U,Uhat,Utmp,Uprev,K,delta,gam1,gam2,gam3,betaDt,bhatDt));
      }

      // FSAL stage
      v = evalRHS(U_halo,K,time + dt);
      maxVelLoc = std::fmax(maxVelLoc,v);
      bhatDt = RKCoeff::bhatfsal*dt;
      Kokkos::parallel_for("RKStageFSAL",cellRng,Numeric::RKFSAL_StageLast_K<decltype(K)>(Uhat,K,bhatDt));

      // Get global max velocity
      Real maxVel = timeStepMaxVelComm(maxVelLoc);

      // Calculate CFL that ended up being used
      Real cfl = dt*maxVel/geom.dmin;

      // Estimate the error and test positivity
      Real errNewLoc = 0.0,posFlag = 1.0;
      const Real nDofs = ps.nX*ps.nY*ps.nZ*ps.nbX*ps.nbY*ps.nbZ*NUM_VARS;
      Kokkos::parallel_reduce("ErrorEstimate",cellRng,
			      Numeric::RKFSAL_ErrEst_K<decltype(U),decltype(K)>
			      (U,Uhat,ps.atol,ps.rtol),errNewLoc,Kokkos::Min<Real>(posFlag));
      Real errNew = timeStepErrEstComm(errNewLoc);
      errNew = 1.0/std::sqrt(errNew/nDofs);

      // Set a new time step size
      Real dtfac = std::pow(errNew,RKCoeff::ep1)*std::pow(errEst,RKCoeff::ep2);
      dtfac = 1.0 + std::atan(dtfac - 1.0);

      // Figure out status of this step
      int lStat,gStat;
      if (posFlag < 0.0) {
	lStat = TSStatus::UNPHYSICAL;
      } else if (dtfac >= ps.rejectionThresh) {
	lStat = TSStatus::ACCEPTED;
      } else {
	lStat = TSStatus::TOLERANCE;
      }
      gStat = timeStepStatusComm(lStat);
      
      if (gStat == TSStatus::UNPHYSICAL) {
	// Solution is unphysical, reject and reduce dt
        PrintSingle(ps,", cfl = %f\n    Rejected: Unphysical\n",cfl);
	// first set to quarter of max cfl, then start halving
	dt = firstUnphys ? 0.25*ps.cfl*geom.dmin/maxVel : dt/2.0;
	firstUnphys = false;
	// Force full weno usage on failure
	useSparseWeno = false;
	nRejectUnphys++;
      } else if (gStat == TSStatus::ACCEPTED) {
	// Step is accepted
        PrintSingle(ps,", cfl = %f\n",cfl);
	accepted = true;
	time += dt;
	Real dterr = dt*dtfac,dtcfl = ps.cfl*geom.dmin/maxVel;
	Real ldt = std::fmin(dterr,dtcfl); // Limit to max cfl
	dt = timeStepSizeComm(ldt); // coordinate all ranks to smallest dt
	errEst = errNew;

	// Update the weno selector
	wenoSelector.update(U,Uprev,wThresh);

	// Move current state to past
	Kokkos::deep_copy(Uprev,U);
	break;
      } else if (gStat == TSStatus::TOLERANCE) {
	// otherwise step is rejected, try again with new smaller dt
        PrintSingle(ps,", cfl = %f\n    Rejected: Tolerance\n",cfl);
	Real ldt = dt*dtfac;
	dt = timeStepSizeComm(ldt); // coordinate all ranks to smallest dt
	nRejectThresh++;
      } else {
	PrintSingle(ps,"Warning: Unknown time step status\n");
	lastTimeStep = true;
	break;
      }
    }

    // Error out if step was rejected too many times or has stagnated
    if (!accepted) {
      PrintSingle(ps,"Warning: Time step was rejected too many times\n");
      lastTimeStep = true;
    }
    if (dt < ps.initialDeltaT) {
      PrintSingle(ps,"Warning: Time step stagnated\n");
      lastTimeStep = true;
    }

    Kokkos::Profiling::popRegion();
  }

  int Solver::timeStepStatusComm(int lStat)
  {
    if (ps.layoutMPI.size == 1) {
      return lStat;
    }
    
    int gStat;
    MPI_Allreduce(&lStat,&gStat,1,MPI_INT,MPI_MAX,ps.layoutMPI.commWorld);
    return gStat;
  }

  Real Solver::timeStepSizeComm(Real ldt)
  {
    if (ps.layoutMPI.size == 1) {
      return ldt;
    }

    Real gdt;
    MPI_Allreduce(&ldt,&gdt,1,ps.layoutMPI.realType,MPI_MIN,ps.layoutMPI.commWorld);
    return gdt;
  }

  Real Solver::timeStepMaxVelComm(Real lV)
  {
    if (ps.layoutMPI.size == 1) {
      return lV;
    }

    Real gV;
    MPI_Allreduce(&lV,&gV,1,ps.layoutMPI.realType,MPI_MAX,ps.layoutMPI.commWorld);
    return gV;
  }

  Real Solver::timeStepErrEstComm(Real lEst)
  {
    if (ps.layoutMPI.size == 1) {
      return lEst;
    }

    Real gEst;
    MPI_Allreduce(&lEst,&gEst,1,ps.layoutMPI.realType,MPI_SUM,ps.layoutMPI.commWorld);
    return gEst;
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
    auto cellRng = interiorCellRange();
    
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
    
    auto cellRng = interiorCellRange();

    if (useSparseWeno) {
      // Do linear reconstruction then weno
      Kokkos::parallel_for("Solver::reconstructRiemannStates(linear)",cellRng,
			   Stencil::KernelLinearRecon_K<decltype(U)>
			   (U,
			    KFVM_D_DECL(faceVals.xDir,
					faceVals.yDir,
					faceVals.zDir),
			    KFVM_D_DECL(stencil.lOff,
					stencil.tOff,
					stencil.ttOff),
			    stencil.faceWeights));
      // Weno reconstruction if needed
      if (wenoSelector.nWeno > 0) {
	auto flagRng = Kokkos::RangePolicy<ExecSpace>
	  (0,wenoSelector.wenoFlagMap.capacity());
	Kokkos::parallel_for("Solver::reconstructRiemannStates(sparse weno)",
			     flagRng,
			     Stencil::KernelWenoRecon_K<decltype(U)>
			     (U,
			      KFVM_D_DECL(ps.nX,ps.nY,ps.nZ),
			      KFVM_D_DECL(wenoSelector.tX,wenoSelector.tY,wenoSelector.tZ),
			      KFVM_D_DECL(faceVals.xDir,
					  faceVals.yDir,
					  faceVals.zDir),
			      wenoSelector.stenWork,
			      wenoSelector.wenoFlagMap,
			      KFVM_D_DECL(stencil.lOff,
					  stencil.tOff,
					  stencil.ttOff),
			      stencil.subIdx,
			      stencil.faceWeights,
			      stencil.derivWeights,
			      ps.fluidProp));
      }
    } else {
      // Weno reconstruction tile by tile
      for (int ntX=0; ntX<wenoSelector.ntX; ntX++) {
	for (int ntY=0; ntY<wenoSelector.ntY; ntY++) {
	  for (int ntZ=0; ntZ<wenoSelector.ntZ; ntZ++) {
	    auto tileRng = wenoSelector.tileRange(ntX,ntY,ntZ);
	    Kokkos::parallel_for("Solver::reconstructRiemannStates(full weno)",
				 tileRng,
				 Stencil::KernelWenoRecon_K<decltype(U)>
				 (U,
				  KFVM_D_DECL(ps.nX,ps.nY,ps.nZ),
				  KFVM_D_DECL(wenoSelector.tX,wenoSelector.tY,wenoSelector.tZ),
				  KFVM_D_DECL(faceVals.xDir,
					      faceVals.yDir,
					      faceVals.zDir),
				  wenoSelector.stenWork,
				  wenoSelector.wenoFlagMap,
				  KFVM_D_DECL(stencil.lOff,
					      stencil.tOff,
					      stencil.ttOff),
				  stencil.subIdx,
				  stencil.faceWeights,
				  stencil.derivWeights,
				  ps.fluidProp));
	  }
	}
      }

      // swap back to sparse weno if allowed
      useSparseWeno = ps.fluidProp.wenoThresh > 0.0;
    }
    
    // Enforce positivity of Riemann states
    Kokkos::parallel_for("PosPres",cellRng,
        		 Physics::PositivityPreserve_K<eqType,decltype(U)>
        		 (U,
        		  KFVM_D_DECL(faceVals.xDir,
        			      faceVals.yDir,
        			      faceVals.zDir),
        		  ps.fluidProp));
  }

  // Set up weno selector and its logic
  WenoSelector::WenoSelector(const ProblemSetup& ps_):
    ps(ps_),
    wenoFlagView("Solver::WenoSelector::wenoFlagView",
		 KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    wenoFlagMap(0),
    nWeno(0)
  {
    // Figure out appropriate tile sizes
    // Want small enough tiles to save memory 
    // unless we are always doing weno
    if (ps.fluidProp.wenoThresh < 0.0) {
      tX = ps.nX; tY = ps.nY; tZ = ps.nZ;
    } else if (SPACE_DIM == 2) {
      tX = (ps.nX > 16 ? ps.nX/8 : 1);
      tY = (ps.nY > 16 ? ps.nY/8 : 1);
      tZ = 1;
    } else {
      tX = (ps.nX > 16 ? ps.nX/4 : 1);
      tY = (ps.nY > 16 ? ps.nY/4 : 1);
      tZ = (ps.nZ > 16 ? ps.nY/4 : 1);
    }

    // count number of tiles
    ntX = ps.nX%tX == 0 ? ps.nX/tX : ps.nX/tX + 1;
    ntY = ps.nY%tY == 0 ? ps.nY/tY : ps.nY/tY + 1;
    ntZ = ps.nZ%tZ == 0 ? ps.nZ/tZ : ps.nZ/tZ + 1;

    // Last tile in any direction could be larger if domain size is weird
    // set minimum allowable bounds accordingly
    idx_t minX = std::max(tX,ps.nX - tX*(ntX - 1));
    idx_t minY = std::max(tY,ps.nY - tY*(ntY - 1));
    idx_t minZ = std::max(tZ,ps.nZ - tZ*(ntZ - 1));
    minSize = minX*minY*minZ;
    currSize = minSize;

    // Allocate everything workspace and map
    wenoFlagMap.rehash(currSize);
    stenWork = Stencil::WorkView("Solver::WenoSelector::stenWork",currSize);
  }

  template<class UViewType>
  void WenoSelector::update(UViewType U,UViewType Uprev,
			    const Real wThresh)
  {
    if (ps.fluidProp.wenoThresh < 0.0) {
      // don't need to manage any of this if weno is used everywhere
      return;
    }
    
    auto cellRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,
					 Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});

    // Set all flags and count how many need weno
    nWeno = 0;
    Kokkos::parallel_reduce("Solver::WenoSelector::update(flag)",
			    cellRng,
			    Numeric::RK_WenoSelect_K<UViewType,
			    decltype(wenoFlagView)>
			    (U,Uprev,ps.fluidProp,wThresh,wenoFlagView),nWeno);

    // Clear map and reallocate workspace if needed
    wenoFlagMap.clear();
    wenoFlagMap.rehash(nWeno);
    if (wenoFlagMap.capacity() > currSize) {
      // Note that capacity != nWeno generally
      PrintAll(ps,"    Realloc workspace from %u to %u on rank %d\n",
	       currSize,wenoFlagMap.capacity(),ps.layoutMPI.rank);
      Kokkos::realloc(stenWork,wenoFlagMap.capacity());
      currSize = wenoFlagMap.capacity();
    }
    
    // Avoid implicit this captures
    auto flagView = wenoFlagView;
    auto flagMap = wenoFlagMap;
    idx_t nX = ps.nX;
#if (SPACE_DIM == 3)
    idx_t nY = ps.nY;
#endif
    // Insert flagged cells
    Kokkos::parallel_for("Solver::WenoSelector::update(insert)",
			 cellRng,
			 KOKKOS_LAMBDA (KFVM_D_DECL(idx_t i,idx_t j,idx_t k)) {
			   if (flagView(KFVM_D_DECL(i,j,k)) > 0) {
#if (SPACE_DIM == 2)
			     idx_t key = nX*j + i;
#else
			     idx_t key = nX*nY*k + nX*j + i;
#endif
			     auto stat = flagMap.insert(key);
			     assert(stat.success());
			   }
			 });
  }
  
  void Solver::setCellBCs(ConsDataView sol_halo,Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::setCellBCs");

    // Do W <=> E comms and set external boundaries
    if (ps.layoutMPI.hasEWComm) {
      commCellBCsEW(sol_halo);
    }
    if (ps.layoutMPI.wDst == MPI_PROC_NULL) {
      setWestBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.eDst == MPI_PROC_NULL) {
      setEastBCExt(sol_halo,t);
    }

    // Do S <=> N comms and set external boundaries
    if (ps.layoutMPI.hasNSComm) {
      commCellBCsNS(sol_halo);
    }
    if (ps.layoutMPI.sDst == MPI_PROC_NULL) {
      setSouthBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.nDst == MPI_PROC_NULL) {
      setNorthBCExt(sol_halo,t);
    }
    
#if (SPACE_DIM == 3)
    // Do B <=> T comms and set external boundaries
    if (ps.layoutMPI.hasTBComm) {
      commCellBCsTB(sol_halo);
    }
    if (ps.layoutMPI.bDst == MPI_PROC_NULL) {
      setBottomBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.tDst == MPI_PROC_NULL) {
      setTopBCExt(sol_halo,t);
    }
#endif
    
    Kokkos::Profiling::popRegion();
  }
  
  void Solver::setFaceBCs(Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::setFaceBCs");

    // Do W <=> E comms and set external boundaries
    if (ps.layoutMPI.hasEWComm) {
      commFaceBCsEW();
    }
    if (ps.layoutMPI.wDst == MPI_PROC_NULL) {
      setWestBCExt(t);
    }
    if (ps.layoutMPI.eDst == MPI_PROC_NULL) {
      setEastBCExt(t);
    }

    // Do S <=> N comms and set external boundaries
    if (ps.layoutMPI.hasNSComm) {
      commFaceBCsNS();
    }
    if (ps.layoutMPI.sDst == MPI_PROC_NULL) {
      setSouthBCExt(t);
    }
    if (ps.layoutMPI.nDst == MPI_PROC_NULL) {
      setNorthBCExt(t);
    }
    
#if (SPACE_DIM == 3)
    // Do B <=> T comms and set external boundaries
    if (ps.layoutMPI.hasTBComm) {
      commFaceBCsTB();
    }
    if (ps.layoutMPI.bDst == MPI_PROC_NULL) {
      setBottomBCExt(t);
    }
    if (ps.layoutMPI.tDst == MPI_PROC_NULL) {
      setTopBCExt(t);
    }
#endif
    
    Kokkos::Profiling::popRegion();
  }

  void Solver::commCellBCsEW(ConsDataView sol_halo)
  {
    // Overall order: pack , sendrecv W -> E, sendrecv E -> W, unpack

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.rad,ps.nY,ps.nZ)});

    // Subviews of sol_halo to pack from/unpack to
    auto wSendSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.wCellSendLo[0],bdyData.wCellSendUp[0]),
							 Kokkos::make_pair(bdyData.wCellSendLo[1],bdyData.wCellSendUp[1]),
							 Kokkos::make_pair(bdyData.wCellSendLo[2],bdyData.wCellSendUp[2])),
				    Kokkos::ALL);
    auto wRecvSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.wCellRecvLo[0],bdyData.wCellRecvUp[0]),
							 Kokkos::make_pair(bdyData.wCellRecvLo[1],bdyData.wCellRecvUp[1]),
							 Kokkos::make_pair(bdyData.wCellRecvLo[2],bdyData.wCellRecvUp[2])),
				    Kokkos::ALL);
    auto eSendSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.eCellSendLo[0],bdyData.eCellSendUp[0]),
							 Kokkos::make_pair(bdyData.eCellSendLo[1],bdyData.eCellSendUp[1]),
							 Kokkos::make_pair(bdyData.eCellSendLo[2],bdyData.eCellSendUp[2])),
				    Kokkos::ALL);
    auto eRecvSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.eCellRecvLo[0],bdyData.eCellRecvUp[0]),
							 Kokkos::make_pair(bdyData.eCellRecvLo[1],bdyData.eCellRecvUp[1]),
							 Kokkos::make_pair(bdyData.eCellRecvLo[2],bdyData.eCellRecvUp[2])),
				    Kokkos::ALL);

    // Communication buffers
    auto wSend = bdyData.wCellSend;
    auto wRecv = bdyData.wCellRecv;
    auto eSend = bdyData.eCellSend;
    auto eRecv = bdyData.eCellRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.wDst == ps.layoutMPI.eSrc) {
      Kokkos::parallel_for("Solver::commCellBCsEW(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       eRecvSub(KFVM_D_DECL(i,j,k),nV) = wSendSub(KFVM_D_DECL(i,j,k),nV);
			       wRecvSub(KFVM_D_DECL(i,j,k),nV) = eSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.wDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsEW(pack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       wSend(KFVM_D_DECL(i,j,k),nV) = wSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.eDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsEW(pack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       eSend(KFVM_D_DECL(i,j,k),nV) = eSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();

    // Communicate W -> E face
    MPI_Sendrecv(wSend.data(),int(wSend.size()),ps.layoutMPI.realType,ps.layoutMPI.wDst,0,
		 eRecv.data(),int(eRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.eSrc,0,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate E -> W face
    MPI_Sendrecv(eSend.data(),int(eSend.size()),ps.layoutMPI.realType,ps.layoutMPI.eDst,1,
		 wRecv.data(),int(wRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.wSrc,1,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.eSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsEW(unpack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       eRecvSub(KFVM_D_DECL(i,j,k),nV) = eRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.wSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsEW(unpack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       wRecvSub(KFVM_D_DECL(i,j,k),nV) = wRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();
  }

  void Solver::commFaceBCsEW()
  {
    // Overall order: recv east, send west, wait, unpack,
    //                recv west, send east, wait, unpack
    
    // MDRange for (un)packing buffers
#if (SPACE_DIM == 2)
    auto bdyRange = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nY});
#else
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nY,ps.nZ});
#endif

    // Subviews of faceVals to pack from/unpack to
    auto wSendSub = Kokkos::subview(faceVals.xDir,
				    KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				    1,Kokkos::ALL,Kokkos::ALL);
    auto wRecvSub = Kokkos::subview(faceVals.xDir,
				    KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				    0,Kokkos::ALL,Kokkos::ALL);
    auto eSendSub = Kokkos::subview(faceVals.xDir,
				    KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				    0,Kokkos::ALL,Kokkos::ALL);
    auto eRecvSub = Kokkos::subview(faceVals.xDir,
				    KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				    1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto wSend = bdyData.wFaceSend;
    auto wRecv = bdyData.wFaceRecv;
    auto eSend = bdyData.eFaceSend;
    auto eRecv = bdyData.eFaceRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.wDst == ps.layoutMPI.eSrc) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 eRecvSub(KFVM_DM_DECL(j,k),nQ,nV) = wSendSub(KFVM_DM_DECL(j,k),nQ,nV);
				 wRecvSub(KFVM_DM_DECL(j,k),nQ,nV) = eSendSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.wDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(pack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 wSend(KFVM_DM_DECL(j,k),nQ,nV) = wSendSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.eDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(pack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 eSend(KFVM_DM_DECL(j,k),nQ,nV) = eSendSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
    
    // Communicate W -> E face
    MPI_Sendrecv(wSend.data(),int(wSend.size()),ps.layoutMPI.realType,ps.layoutMPI.wDst,2,
		 eRecv.data(),int(eRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.eSrc,2,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate E -> W face
    MPI_Sendrecv(eSend.data(),int(eSend.size()),ps.layoutMPI.realType,ps.layoutMPI.eDst,3,
		 wRecv.data(),int(wRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.wSrc,3,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.eSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(unpack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 eRecvSub(KFVM_DM_DECL(j,k),nQ,nV) = eRecv(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.wSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(unpack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 wRecvSub(KFVM_DM_DECL(j,k),nQ,nV) = wRecv(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
  }

  void Solver::commCellBCsNS(ConsDataView sol_halo)
  {
    // Overall order: send south, recv north, send north, recv south

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ)});

    // Subviews of sol_halo to pack from/unpack to
    auto sSendSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.sCellSendLo[0],bdyData.sCellSendUp[0]),
							 Kokkos::make_pair(bdyData.sCellSendLo[1],bdyData.sCellSendUp[1]),
							 Kokkos::make_pair(bdyData.sCellSendLo[2],bdyData.sCellSendUp[2])),
				    Kokkos::ALL);
    auto sRecvSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.sCellRecvLo[0],bdyData.sCellRecvUp[0]),
							 Kokkos::make_pair(bdyData.sCellRecvLo[1],bdyData.sCellRecvUp[1]),
							 Kokkos::make_pair(bdyData.sCellRecvLo[2],bdyData.sCellRecvUp[2])),
				    Kokkos::ALL);
    auto nSendSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.nCellSendLo[0],bdyData.nCellSendUp[0]),
							 Kokkos::make_pair(bdyData.nCellSendLo[1],bdyData.nCellSendUp[1]),
							 Kokkos::make_pair(bdyData.nCellSendLo[2],bdyData.nCellSendUp[2])),
				    Kokkos::ALL);
    auto nRecvSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.nCellRecvLo[0],bdyData.nCellRecvUp[0]),
							 Kokkos::make_pair(bdyData.nCellRecvLo[1],bdyData.nCellRecvUp[1]),
							 Kokkos::make_pair(bdyData.nCellRecvLo[2],bdyData.nCellRecvUp[2])),
				    Kokkos::ALL);

    // Communication buffers
    auto sSend = bdyData.sCellSend;
    auto sRecv = bdyData.sCellRecv;
    auto nSend = bdyData.nCellSend;
    auto nRecv = bdyData.nCellRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.sDst == ps.layoutMPI.nSrc) {
      Kokkos::parallel_for("Solver::commCellBCsNS(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       nRecvSub(KFVM_D_DECL(i,j,k),nV) = sSendSub(KFVM_D_DECL(i,j,k),nV);
			       sRecvSub(KFVM_D_DECL(i,j,k),nV) = nSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.sDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(pack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       sSend(KFVM_D_DECL(i,j,k),nV) = sSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.nDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(pack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       nSend(KFVM_D_DECL(i,j,k),nV) = nSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();

    // Communicate S -> N face
    MPI_Sendrecv(sSend.data(),int(sSend.size()),ps.layoutMPI.realType,ps.layoutMPI.sDst,0,
		 nRecv.data(),int(nRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.nSrc,0,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate N -> S face
    MPI_Sendrecv(nSend.data(),int(nSend.size()),ps.layoutMPI.realType,ps.layoutMPI.nDst,1,
		 sRecv.data(),int(sRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.sSrc,1,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.nSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(unpack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       nRecvSub(KFVM_D_DECL(i,j,k),nV) = nRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.sSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(unpack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       sRecvSub(KFVM_D_DECL(i,j,k),nV) = sRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();
  }

  void Solver::commFaceBCsNS()
  {
    // Overall order: send south, recv north, send north, recv south
    // MDRange for (un)packing buffers
#if (SPACE_DIM == 2)
    auto bdyRange = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nX});
#else
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM - 1>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nZ});
#endif

    // Subviews of faceVals to pack from/unpack to
    auto sSendSub = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				    1,Kokkos::ALL,Kokkos::ALL);
    auto sRecvSub = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				    0,Kokkos::ALL,Kokkos::ALL);
    auto nSendSub = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				    0,Kokkos::ALL,Kokkos::ALL);
    auto nRecvSub = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				    1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto sSend = bdyData.sFaceSend;
    auto sRecv = bdyData.sFaceRecv;
    auto nSend = bdyData.nFaceSend;
    auto nRecv = bdyData.nFaceRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.sDst == ps.layoutMPI.nSrc) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 nRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = sSendSub(KFVM_DM_DECL(i,k),nQ,nV);
				 sRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = nSendSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.sDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(pack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 sSend(KFVM_DM_DECL(i,k),nQ,nV) = sSendSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.nDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(pack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 nSend(KFVM_DM_DECL(i,k),nQ,nV) = nSendSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
    
    // Communicate S -> N face
    MPI_Sendrecv(sSend.data(),int(sSend.size()),ps.layoutMPI.realType,ps.layoutMPI.sDst,2,
		 nRecv.data(),int(nRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.nSrc,2,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate N -> S face
    MPI_Sendrecv(nSend.data(),int(nSend.size()),ps.layoutMPI.realType,ps.layoutMPI.nDst,3,
		 sRecv.data(),int(sRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.sSrc,3,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.nSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(unpack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 nRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = nRecv(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.sSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(unpack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 sRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = sRecv(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
  }

#if (SPACE_DIM == 3)
  void Solver::commCellBCsTB(ConsDataView sol_halo)
  {
    // Overall order: send south, recv north, send north, recv south

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({0,0,0},{ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad});

    // Subviews of sol_halo to pack from/unpack to
    auto bSendSub = Kokkos::subview(sol_halo,
				    Kokkos::make_pair(bdyData.bCellSendLo[0],bdyData.bCellSendUp[0]),
				    Kokkos::make_pair(bdyData.bCellSendLo[1],bdyData.bCellSendUp[1]),
				    Kokkos::make_pair(bdyData.bCellSendLo[2],bdyData.bCellSendUp[2]),
				    Kokkos::ALL);
    auto bRecvSub = Kokkos::subview(sol_halo,
				    Kokkos::make_pair(bdyData.bCellRecvLo[0],bdyData.bCellRecvUp[0]),
				    Kokkos::make_pair(bdyData.bCellRecvLo[1],bdyData.bCellRecvUp[1]),
				    Kokkos::make_pair(bdyData.bCellRecvLo[2],bdyData.bCellRecvUp[2]),
				    Kokkos::ALL);
    auto tSendSub = Kokkos::subview(sol_halo,
				    Kokkos::make_pair(bdyData.tCellSendLo[0],bdyData.tCellSendUp[0]),
				    Kokkos::make_pair(bdyData.tCellSendLo[1],bdyData.tCellSendUp[1]),
				    Kokkos::make_pair(bdyData.tCellSendLo[2],bdyData.tCellSendUp[2]),
				    Kokkos::ALL);
    auto tRecvSub = Kokkos::subview(sol_halo,
				    Kokkos::make_pair(bdyData.tCellRecvLo[0],bdyData.tCellRecvUp[0]),
				    Kokkos::make_pair(bdyData.tCellRecvLo[1],bdyData.tCellRecvUp[1]),
				    Kokkos::make_pair(bdyData.tCellRecvLo[2],bdyData.tCellRecvUp[2]),
				    Kokkos::ALL);

    // Communication buffers
    auto bSend = bdyData.bCellSend;
    auto bRecv = bdyData.bCellRecv;
    auto tSend = bdyData.tCellSend;
    auto tRecv = bdyData.tCellRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.bDst == ps.layoutMPI.tSrc) {
      Kokkos::parallel_for("Solver::commCellBCsTB(copy)",bdyRange,
			   KOKKOS_LAMBDA (const idx_t i,const idx_t j,const idx_t k) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       tRecvSub(i,j,k,nV) = bSendSub(i,j,k,nV);
			       bRecvSub(i,j,k,nV) = tSendSub(i,j,k,nV);
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.bDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(pack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       bSend(KFVM_D_DECL(i,j,k),nV) = bSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.tDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(pack top)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       tSend(KFVM_D_DECL(i,j,k),nV) = tSendSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();

    // Communicate B -> T face
     MPI_Sendrecv(bSend.data(),int(bSend.size()),ps.layoutMPI.realType,ps.layoutMPI.bDst,0,
		  tRecv.data(),int(tRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.tSrc,0,
		  ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate T -> B face
    MPI_Sendrecv(tSend.data(),int(tSend.size()),ps.layoutMPI.realType,ps.layoutMPI.tDst,1,
		 bRecv.data(),int(bRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.bSrc,1,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.tSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(unpack top)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       tRecvSub(KFVM_D_DECL(i,j,k),nV) = tRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    if (ps.layoutMPI.bSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commCellBCsNS(unpack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       bRecvSub(KFVM_D_DECL(i,j,k),nV) = bRecv(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
    Kokkos::fence();
  }

  void Solver::commFaceBCsTB()
  {
    // Overall order: send bottom, recv top, send top, recv bottom
    // Range for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM - 1>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nY});

    // Subviews of faceVals to pack from/unpack to
    auto bSendSub = Kokkos::subview(faceVals.zDir,
				    Kokkos::ALL,Kokkos::ALL,0,
				    1,Kokkos::ALL,Kokkos::ALL);
    auto bRecvSub = Kokkos::subview(faceVals.zDir,
				    Kokkos::ALL,Kokkos::ALL,0,
				    0,Kokkos::ALL,Kokkos::ALL);
    auto tSendSub = Kokkos::subview(faceVals.zDir,
				    Kokkos::ALL,Kokkos::ALL,ps.nZ,
				    0,Kokkos::ALL,Kokkos::ALL);
    auto tRecvSub = Kokkos::subview(faceVals.zDir,
				    Kokkos::ALL,Kokkos::ALL,ps.nZ,
				    1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto bSend = bdyData.bFaceSend;
    auto bRecv = bdyData.bFaceRecv;
    auto tSend = bdyData.tFaceSend;
    auto tRecv = bdyData.tFaceRecv;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.bDst == ps.layoutMPI.tSrc) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t j)) {
			     const idx_t nQuad = NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 tRecvSub(i,j,nQ,nV) = bSendSub(i,j,nQ,nV);
				 bRecvSub(i,j,nQ,nV) = tSendSub(i,j,nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack send buffers
    if (ps.layoutMPI.bDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(pack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 bSend(KFVM_DM_DECL(i,k),nQ,nV) = bSendSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.tDst != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(pack top)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 tSend(KFVM_DM_DECL(i,k),nQ,nV) = tSendSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
    
    // Communicate T -> B face
    MPI_Sendrecv(bSend.data(),int(bSend.size()),ps.layoutMPI.realType,ps.layoutMPI.bDst,2,
		 tRecv.data(),int(tRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.tSrc,2,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Communicate T -> B face
    MPI_Sendrecv(tSend.data(),int(tSend.size()),ps.layoutMPI.realType,ps.layoutMPI.tDst,3,
		 bRecv.data(),int(bRecv.size()),ps.layoutMPI.realType,ps.layoutMPI.bSrc,3,
		 ps.layoutMPI.commWorld,MPI_STATUS_IGNORE);
    
    // Unpack recv buffers
    if (ps.layoutMPI.tSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(unpack top)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 tRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = tRecv(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    if (ps.layoutMPI.bSrc != MPI_PROC_NULL) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(unpack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 bRecvSub(KFVM_DM_DECL(i,k),nQ,nV) = bRecv(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
    Kokkos::fence();
  }
#endif

  void Solver::setWestBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcWest_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad + ps.nY});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad},{ps.nY + ps.rad,ps.nZ + ps.rad});
#endif
    
    switch (ps.bcType[FaceLabel::west]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::West",bndRng,
			   CellBcWest_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::West",bndRng,
			   CellBcWest_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.x)>
			   (sol_halo,bcCoeff.x,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::West",bndRng,
			   CellBcWest_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nX,t));
      break;
    default:
      PrintSingle(ps,"Warning: Western cell BC undefined.\n");
    }
  }

  void Solver::setEastBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcEast_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad + ps.nY});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({ps.rad,ps.rad},{ps.nY + ps.rad,ps.nZ + ps.rad});
#endif

    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::East",bndRng,
			   CellBcEast_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nX));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::East",bndRng,
			   CellBcEast_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.x)>
			   (sol_halo,bcCoeff.x,ps.rad,ps.nX));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::East",bndRng,
			   CellBcEast_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nX,t));
      break;
    default:
      PrintSingle(ps,"Warning: Eastern cell BC undefined.\n");
    }    
  }

  void Solver::setSouthBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcSouth_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({0,ps.nX + 2*ps.rad});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,ps.rad},{ps.nX + 2*ps.rad,ps.nZ + ps.rad});
#endif

    switch (ps.bcType[FaceLabel::south]) {
    case BCType::outflow :
      Kokkos::parallel_for("CellBCs::South",bndRng,
			   CellBcSouth_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::reflecting :
      Kokkos::parallel_for("CellBCs::South",bndRng,
			   CellBcSouth_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.y)>
			   (sol_halo,bcCoeff.y,ps.rad,ps.nY));
      break;
    case BCType::user :
      Kokkos::parallel_for("CellBCs::South",bndRng,
			   CellBcSouth_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nY,t));
      break;
    default:
      PrintSingle(ps,"Warning: Southern cell BC undefined.\n");
    }    
  }

  void Solver::setNorthBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcNorth_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>
      ({0,ps.nX + 2*ps.rad});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,ps.rad},{ps.nX + 2*ps.rad,ps.nZ + ps.rad});
#endif

    switch (ps.bcType[FaceLabel::north]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::North",bndRng,
			   CellBcNorth_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nY));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::North",bndRng,
			   CellBcNorth_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.y)>
			   (sol_halo,bcCoeff.y,ps.rad,ps.nY));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::North",bndRng,
			   CellBcNorth_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nY,t));
      break;
    default:
      PrintSingle(ps,"Warning: Northern cell BC undefined.\n");
    }    
  }

#if (SPACE_DIM == 3)
  void Solver::setBottomBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcBottom_K;
    
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX + 2*ps.rad,ps.nY + 2*ps.rad});
    
    switch (ps.bcType[FaceLabel::bottom]) {
    case BCType::outflow :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng,
			   CellBcBottom_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::reflecting :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng,
			   CellBcBottom_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.z)>
			   (sol_halo,bcCoeff.z,ps.rad,ps.nZ));
      break;
    case BCType::user :
      Kokkos::parallel_for("CellBCs::Bottom",bndRng,
			   CellBcBottom_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nZ,t));
      break;
    default:
      PrintSingle(ps,"Warning: Bottom cell BC undefined.\n");
    }
  }
  
  void Solver::setTopBCExt(ConsDataView sol_halo,Real t)
  {
    using BoundaryConditions::CellBcTop_K;
    
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX + 2*ps.rad,ps.nY + 2*ps.rad});

    switch (ps.bcType[FaceLabel::top]) {
    case BCType::outflow:
      Kokkos::parallel_for("CellBCs::Top",bndRng,
			   CellBcTop_K<decltype(sol_halo),BCType::outflow>
			   (sol_halo,ps.rad,ps.nZ));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("CellBCs::Top",bndRng,
			   CellBcTop_K<decltype(sol_halo),BCType::reflecting,decltype(bcCoeff.z)>
			   (sol_halo,bcCoeff.z,ps.rad,ps.nZ));
      break;
    case BCType::user:
      Kokkos::parallel_for("CellBCs::Top",bndRng,
			   CellBcTop_K<decltype(sol_halo),BCType::user>
			   (sol_halo,geom,ps.rad,ps.nZ,t));
      break;
    default:
      PrintSingle(ps,"Warning: Top cell BC undefined.\n");
    }
  }
#endif

  void Solver::setWestBCExt(Real t)
  {
    // Western Boundary
    using BoundaryConditions::FaceBcWest_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nY});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nY,ps.nZ});
#endif
    
    auto westBnd = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::west]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::West::Outflow",bndRng,
			   FaceBcWest_K<decltype(westBnd),BCType::outflow>(westBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::West::Reflecting",bndRng,
			   FaceBcWest_K<decltype(westBnd),BCType::reflecting,decltype(bcCoeff.x)>(westBnd,bcCoeff.x));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::West::User",bndRng,
			   FaceBcWest_K<decltype(westBnd),BCType::user>(westBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Western face BC undefined.\n");
    }
  }
  
  void Solver::setEastBCExt(Real t)
  {
    // Eastern Boundary
    using BoundaryConditions::FaceBcEast_K;
    
#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nY});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nY,ps.nZ});
#endif
    
    auto eastBnd = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				   Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::east]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::East::Outflow",bndRng,
			   FaceBcEast_K<decltype(eastBnd),BCType::outflow>(eastBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::East::Reflecting",bndRng,
			   FaceBcEast_K<decltype(eastBnd),BCType::reflecting,decltype(bcCoeff.x)>(eastBnd,bcCoeff.x));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::East::User",bndRng,
			   FaceBcEast_K<decltype(eastBnd),BCType::user>(eastBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Eastern face BC undefined.\n");
    }
  }
  
  void Solver::setSouthBCExt(Real t)
  {
    // Southern Boundary
    using BoundaryConditions::FaceBcSouth_K;

#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nX});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nZ});
#endif
    
    auto southBnd = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::south]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::South::Outflow",bndRng,
			   FaceBcSouth_K<decltype(southBnd),BCType::outflow>(southBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::South::Reflecting",bndRng,
			   FaceBcSouth_K<decltype(southBnd),BCType::reflecting,decltype(bcCoeff.y)>(southBnd,bcCoeff.y));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::South::User",bndRng,
			   FaceBcSouth_K<decltype(southBnd),BCType::user>(southBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Southern face BC undefined.\n");
    }
  }
  
  void Solver::setNorthBCExt(Real t)
  {
    // Northern Boundary
    using BoundaryConditions::FaceBcNorth_K;

#if (SPACE_DIM == 2)
    auto bndRng = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nX});
#else
    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nZ});
#endif
    
    auto northBnd = Kokkos::subview(faceVals.yDir,
				    KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				    Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::north]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::North::Outflow",bndRng,
			   FaceBcNorth_K<decltype(northBnd),BCType::outflow>(northBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::North::Reflecting",bndRng,
			   FaceBcNorth_K<decltype(northBnd),BCType::reflecting,decltype(bcCoeff.y)>(northBnd,bcCoeff.y));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::North::User",bndRng,
			   FaceBcNorth_K<decltype(northBnd),BCType::user>(northBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Northern face BC undefined.\n");
    }
  }

#if (SPACE_DIM == 3)  
  void Solver::setBottomBCExt(Real t)
  {
    // Bottom Boundary
    using BoundaryConditions::FaceBcBottom_K;

    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nY});
    
    auto bottomBnd = Kokkos::subview(faceVals.zDir,
				     Kokkos::ALL,Kokkos::ALL,0,
				     Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::bottom]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Bottom::Outflow",bndRng,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::outflow>(bottomBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Bottom::Reflecting",bndRng,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::reflecting,decltype(bcCoeff.z)>(bottomBnd,bcCoeff.z));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Bottom::User",bndRng,
			   FaceBcBottom_K<decltype(bottomBnd),BCType::user>(bottomBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Bottom face BC undefined.\n");
    }
  }
  
  void Solver::setTopBCExt(Real t)
  {
    // Top Boundary
    using BoundaryConditions::FaceBcTop_K;

    auto bndRng = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<2>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nY});
    
    auto topBnd = Kokkos::subview(faceVals.zDir,
				  Kokkos::ALL,Kokkos::ALL,ps.nZ,
				  Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
    
    switch (ps.bcType[FaceLabel::top]) {
    case BCType::outflow:
      Kokkos::parallel_for("FaceBCs::Top::Outflow",bndRng,
			   FaceBcTop_K<decltype(topBnd),BCType::outflow>(topBnd));
      break;
    case BCType::reflecting:
      Kokkos::parallel_for("FaceBCs::Top::Reflecting",bndRng,
			   FaceBcTop_K<decltype(topBnd),BCType::reflecting,decltype(bcCoeff.z)>(topBnd,bcCoeff.z));
      break;
    case BCType::user:
      Kokkos::parallel_for("FaceBCs::Top::User",bndRng,
			   FaceBcTop_K<decltype(topBnd),BCType::user>(topBnd,geom,qr.ab,t));
      break;
    default:
      PrintSingle(ps,"Warning: Top face BC undefined.\n");
    }
  }
#endif

  void Solver::setIC()
  {
    Kokkos::Profiling::pushRegion("Solver::setIC");

    // Fill U using user specified initial condition
    auto U = trimCellHalo(U_halo);
    auto cellRng = interiorCellRange();
    Kokkos::parallel_for("IntegrateIC",cellRng,Numeric::IntegrateIC_K<decltype(U)>
			 (U,qr.ab,qr.wt,geom));

    // Set Uprev to also be IC
    // Needed for error control in timestepper
    Kokkos::deep_copy(Uprev_halo,U_halo);

    Kokkos::Profiling::popRegion();
  }
  
} // end namespace KFVM
