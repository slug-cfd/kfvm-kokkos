// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

#include <array>
#include <cstdint>
#include <cstdio>
#include <limits>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "BoundaryData.H"
#include "Dimension.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "Geometry.H"
#include "BoundaryConditions_K.H"
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
    bdyData(ps),
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
    writerPDI.write(U_halo,U_aux,wenoSelect,0,time);

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
        writerPDI.write(U_halo,U_aux,wenoSelect,nT,time);
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
	Kokkos::parallel_for("WenoSelector",cellRng,
                             Numeric::RK_WenoSelect_K<decltype(U),decltype(wenoSelect)>(U,Uprev,wenoSelect,ps.fluidProp));
	
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
    Kokkos::Profiling::pushRegion("Solver::setCellBCs");

    // Do W <=> E comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::east] >= 0) {
      commCellBCsEW(sol_halo);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::west] == -1) {
      setWestBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::east] == -1) {
      setEastBCExt(sol_halo,t);
    }    

    // Do S <=> N comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::north] >= 0) {
      commCellBCsNS(sol_halo);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::south] == -1) {
      setSouthBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::north] == -1) {
      setNorthBCExt(sol_halo,t);
    }
    
#if (SPACE_DIM == 3)
    // Do B <=> T comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::top] >= 0) {
      commCellBCsTB(sol_halo);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] == -1) {
      setBottomBCExt(sol_halo,t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::top] == -1) {
      setTopBCExt(sol_halo,t);
    }
#endif
    
    Kokkos::Profiling::popRegion();
  }
  
  void Solver::setFaceBCs(Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::setFaceBCs");

    // Do W <=> E comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::east] >= 0) {
      commFaceBCsEW();
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::west] == -1) {
      setWestBCExt(t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::east] == -1) {
      setEastBCExt(t);
    }    

    // Do S <=> N comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::north] >= 0) {
      commFaceBCsNS();
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::south] == -1) {
      setSouthBCExt(t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::north] == -1) {
      setNorthBCExt(t);
    }
    
#if (SPACE_DIM == 3)
    // Do B <=> T comms and set external boundaries
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0 ||
	ps.layoutMPI.nbrRank[FaceLabel::top] >= 0) {
      commFaceBCsTB();
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] == -1) {
      setBottomBCExt(t);
    }
    if (ps.layoutMPI.nbrRank[FaceLabel::top] == -1) {
      setTopBCExt(t);
    }
#endif
    
    Kokkos::Profiling::popRegion();
  }

  void Solver::commCellBCsEW(ConsDataView sol_halo)
  {
    // Overall order: send west, recv east, send east, recv west

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.rad,ps.nY,ps.nZ)});

    // Subviews of sol_halo to pack from/unpack to
    auto wSrcSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.wCellSrcLo[0],bdyData.wCellSrcUp[0]),
							Kokkos::make_pair(bdyData.wCellSrcLo[1],bdyData.wCellSrcUp[1]),
							Kokkos::make_pair(bdyData.wCellSrcLo[2],bdyData.wCellSrcUp[2])),
				   Kokkos::ALL);
    auto wDstSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.wCellDstLo[0],bdyData.wCellDstUp[0]),
							Kokkos::make_pair(bdyData.wCellDstLo[1],bdyData.wCellDstUp[1]),
							Kokkos::make_pair(bdyData.wCellDstLo[2],bdyData.wCellDstUp[2])),
				   Kokkos::ALL);
    auto eSrcSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.eCellSrcLo[0],bdyData.eCellSrcUp[0]),
							Kokkos::make_pair(bdyData.eCellSrcLo[1],bdyData.eCellSrcUp[1]),
							Kokkos::make_pair(bdyData.eCellSrcLo[2],bdyData.eCellSrcUp[2])),
				   Kokkos::ALL);
    auto eDstSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.eCellDstLo[0],bdyData.eCellDstUp[0]),
							Kokkos::make_pair(bdyData.eCellDstLo[1],bdyData.eCellDstUp[1]),
							Kokkos::make_pair(bdyData.eCellDstLo[2],bdyData.eCellDstUp[2])),
				   Kokkos::ALL);

    // Communication buffers
    auto wSrc = bdyData.wCellSrc;
    auto wDst = bdyData.wCellDst;
    auto eSrc = bdyData.eCellSrc;
    auto eDst = bdyData.eCellDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::west] == ps.layoutMPI.nbrRank[FaceLabel::east]) {
      Kokkos::parallel_for("Solver::commCellBCsEW(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       eDstSub(KFVM_D_DECL(i,j,k),nV) = wSrcSub(KFVM_D_DECL(i,j,k),nV);
			       wDstSub(KFVM_D_DECL(i,j,k),nV) = eSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      return;
    }
    
    // Pack and send west src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0) {
      Kokkos::parallel_for("Solver::commCellBCsEW(pack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       wSrc(KFVM_D_DECL(i,j,k),nV) = wSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      
      MPI_Send(wSrc.data(),int(wSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::west],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv east buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::east] >= 0) {
      MPI_Recv(eDst.data(),int(eDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::east],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsEW(pack/unpack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       eDstSub(KFVM_D_DECL(i,j,k),nV) = eDst(KFVM_D_DECL(i,j,k),nV);
			       eSrc(KFVM_D_DECL(i,j,k),nV) = eSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      
      MPI_Send(eSrc.data(),int(eSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::east],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack west buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0) {
      MPI_Recv(wDst.data(),int(wDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::west],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsEW(pack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       wDstSub(KFVM_D_DECL(i,j,k),nV) = wDst(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
  }

  void Solver::commFaceBCsEW()
  {
    // Overall order: send west, recv east, send east, recv west

    // MDRange for (un)packing buffers
#if (SPACE_DIM == 2)
    auto bdyRange = Kokkos::RangePolicy<ExecSpace,Kokkos::IndexType<idx_t>>({0,ps.nY});
#else
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM - 1>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nY,ps.nZ});
#endif

    // Subviews of faceVals to pack from/unpack to
    auto wSrcSub = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				   1,Kokkos::ALL,Kokkos::ALL);
    auto wDstSub = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(0,Kokkos::ALL,Kokkos::ALL),
				   0,Kokkos::ALL,Kokkos::ALL);
    auto eSrcSub = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				   0,Kokkos::ALL,Kokkos::ALL);
    auto eDstSub = Kokkos::subview(faceVals.xDir,
				   KFVM_D_DECL(ps.nX,Kokkos::ALL,Kokkos::ALL),
				   1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto wSrc = bdyData.wFaceSrc;
    auto wDst = bdyData.wFaceDst;
    auto eSrc = bdyData.eFaceSrc;
    auto eDst = bdyData.eFaceDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::west] == ps.layoutMPI.nbrRank[FaceLabel::east]) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 eDstSub(KFVM_DM_DECL(j,k),nQ,nV) = wSrcSub(KFVM_DM_DECL(j,k),nQ,nV);
				 wDstSub(KFVM_DM_DECL(j,k),nQ,nV) = eSrcSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack and send west src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0) {
      Kokkos::parallel_for("Solver::commFaceBCsEW(pack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 wSrc(KFVM_DM_DECL(j,k),nQ,nV) = wSrcSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(wSrc.data(),int(wSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::west],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv east buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::east] >= 0) {
      MPI_Recv(eDst.data(),int(eDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::east],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsEW(pack/unpack east)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 eDstSub(KFVM_DM_DECL(j,k),nQ,nV) = eDst(KFVM_DM_DECL(j,k),nQ,nV);
				 eSrc(KFVM_DM_DECL(j,k),nQ,nV) = eSrcSub(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(eSrc.data(),int(eSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::east],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack west buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::west] >= 0) {
      MPI_Recv(wDst.data(),int(wDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::west],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsEW(unpack west)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t j,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 wDstSub(KFVM_DM_DECL(j,k),nQ,nV) = wDst(KFVM_DM_DECL(j,k),nQ,nV);
			       }
			     }
			   });
    }
  }

  void Solver::commCellBCsNS(ConsDataView sol_halo)
  {
    // Overall order: send south, recv north, send north, recv south

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX + 2*ps.rad,ps.rad,ps.nZ)});

    // Subviews of sol_halo to pack from/unpack to
    auto sSrcSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.sCellSrcLo[0],bdyData.sCellSrcUp[0]),
							Kokkos::make_pair(bdyData.sCellSrcLo[1],bdyData.sCellSrcUp[1]),
							Kokkos::make_pair(bdyData.sCellSrcLo[2],bdyData.sCellSrcUp[2])),
				   Kokkos::ALL);
    auto sDstSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.sCellDstLo[0],bdyData.sCellDstUp[0]),
							Kokkos::make_pair(bdyData.sCellDstLo[1],bdyData.sCellDstUp[1]),
							Kokkos::make_pair(bdyData.sCellDstLo[2],bdyData.sCellDstUp[2])),
				   Kokkos::ALL);
    auto nSrcSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.nCellSrcLo[0],bdyData.nCellSrcUp[0]),
							Kokkos::make_pair(bdyData.nCellSrcLo[1],bdyData.nCellSrcUp[1]),
							Kokkos::make_pair(bdyData.nCellSrcLo[2],bdyData.nCellSrcUp[2])),
				   Kokkos::ALL);
    auto nDstSub = Kokkos::subview(sol_halo,KFVM_D_DECL(Kokkos::make_pair(bdyData.nCellDstLo[0],bdyData.nCellDstUp[0]),
							Kokkos::make_pair(bdyData.nCellDstLo[1],bdyData.nCellDstUp[1]),
							Kokkos::make_pair(bdyData.nCellDstLo[2],bdyData.nCellDstUp[2])),
				   Kokkos::ALL);

    // Communication buffers
    auto sSrc = bdyData.sCellSrc;
    auto sDst = bdyData.sCellDst;
    auto nSrc = bdyData.nCellSrc;
    auto nDst = bdyData.nCellDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::south] == ps.layoutMPI.nbrRank[FaceLabel::north]) {
      Kokkos::parallel_for("Solver::commCellBCsNS(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       nDstSub(KFVM_D_DECL(i,j,k),nV) = sSrcSub(KFVM_D_DECL(i,j,k),nV);
			       sDstSub(KFVM_D_DECL(i,j,k),nV) = nSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      return;
    }
    
    // Pack and send south src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0) {
      Kokkos::parallel_for("Solver::commCellBCsNS(pack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       sSrc(KFVM_D_DECL(i,j,k),nV) = sSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      
      MPI_Send(sSrc.data(),int(sSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::south],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv north buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::north] >= 0) {
      MPI_Recv(nDst.data(),int(nDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::north],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsNS(pack/unpack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       nDstSub(KFVM_D_DECL(i,j,k),nV) = nDst(KFVM_D_DECL(i,j,k),nV);
			       nSrc(KFVM_D_DECL(i,j,k),nV) = nSrcSub(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
      
      MPI_Send(nSrc.data(),int(nSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::north],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack south buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0) {
      MPI_Recv(sDst.data(),int(sDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::south],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsNS(pack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       sDstSub(KFVM_D_DECL(i,j,k),nV) = sDst(KFVM_D_DECL(i,j,k),nV);
			     }
			   });
    }
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
    auto sSrcSub = Kokkos::subview(faceVals.yDir,
				   KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				   1,Kokkos::ALL,Kokkos::ALL);
    auto sDstSub = Kokkos::subview(faceVals.yDir,
				   KFVM_D_DECL(Kokkos::ALL,0,Kokkos::ALL),
				   0,Kokkos::ALL,Kokkos::ALL);
    auto nSrcSub = Kokkos::subview(faceVals.yDir,
				   KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				   0,Kokkos::ALL,Kokkos::ALL);
    auto nDstSub = Kokkos::subview(faceVals.yDir,
				   KFVM_D_DECL(Kokkos::ALL,ps.nY,Kokkos::ALL),
				   1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto sSrc = bdyData.sFaceSrc;
    auto sDst = bdyData.sFaceDst;
    auto nSrc = bdyData.nFaceSrc;
    auto nDst = bdyData.nFaceDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::south] == ps.layoutMPI.nbrRank[FaceLabel::north]) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 nDstSub(KFVM_DM_DECL(i,k),nQ,nV) = sSrcSub(KFVM_DM_DECL(i,k),nQ,nV);
				 sDstSub(KFVM_DM_DECL(i,k),nQ,nV) = nSrcSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack and send south src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0) {
      Kokkos::parallel_for("Solver::commFaceBCsNS(pack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 sSrc(KFVM_DM_DECL(i,k),nQ,nV) = sSrcSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(sSrc.data(),int(sSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::south],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv north buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::north] >= 0) {
      MPI_Recv(nDst.data(),int(nDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::north],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsNS(pack/unpack north)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 nDstSub(KFVM_DM_DECL(i,k),nQ,nV) = nDst(KFVM_DM_DECL(i,k),nQ,nV);
				 nSrc(KFVM_DM_DECL(i,k),nQ,nV) = nSrcSub(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(nSrc.data(),int(nSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::north],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack south buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::south] >= 0) {
      MPI_Recv(sDst.data(),int(sDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::south],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsNS(unpack south)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t k)) {
			     const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 sDstSub(KFVM_DM_DECL(i,k),nQ,nV) = sDst(KFVM_DM_DECL(i,k),nQ,nV);
			       }
			     }
			   });
    }
  }

#if (SPACE_DIM == 3)
  void Solver::commCellBCsTB(ConsDataView sol_halo)
  {
    // Overall order: send south, recv north, send north, recv south

    // MDRange for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM>,Kokkos::IndexType<idx_t>>
      ({0,0,0},{ps.nX + 2*ps.rad,ps.nY + 2*ps.rad,ps.rad});

    // Subviews of sol_halo to pack from/unpack to
    auto bSrcSub = Kokkos::subview(sol_halo,
				   Kokkos::make_pair(bdyData.bCellSrcLo[0],bdyData.bCellSrcUp[0]),
				   Kokkos::make_pair(bdyData.bCellSrcLo[1],bdyData.bCellSrcUp[1]),
				   Kokkos::make_pair(bdyData.bCellSrcLo[2],bdyData.bCellSrcUp[2]),
				   Kokkos::ALL);
    auto bDstSub = Kokkos::subview(sol_halo,
				   Kokkos::make_pair(bdyData.bCellDstLo[0],bdyData.bCellDstUp[0]),
				   Kokkos::make_pair(bdyData.bCellDstLo[1],bdyData.bCellDstUp[1]),
				   Kokkos::make_pair(bdyData.bCellDstLo[2],bdyData.bCellDstUp[2]),
				   Kokkos::ALL);
    auto tSrcSub = Kokkos::subview(sol_halo,
				   Kokkos::make_pair(bdyData.tCellSrcLo[0],bdyData.tCellSrcUp[0]),
				   Kokkos::make_pair(bdyData.tCellSrcLo[1],bdyData.tCellSrcUp[1]),
				   Kokkos::make_pair(bdyData.tCellSrcLo[2],bdyData.tCellSrcUp[2]),
				   Kokkos::ALL);
    auto tDstSub = Kokkos::subview(sol_halo,
				   Kokkos::make_pair(bdyData.tCellDstLo[0],bdyData.tCellDstUp[0]),
				   Kokkos::make_pair(bdyData.tCellDstLo[1],bdyData.tCellDstUp[1]),
				   Kokkos::make_pair(bdyData.tCellDstLo[2],bdyData.tCellDstUp[2]),
				   Kokkos::ALL);

    // Communication buffers
    auto bSrc = bdyData.bCellSrc;
    auto bDst = bdyData.bCellDst;
    auto tSrc = bdyData.tCellSrc;
    auto tDst = bdyData.tCellDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] == ps.layoutMPI.nbrRank[FaceLabel::top]) {
      Kokkos::parallel_for("Solver::commCellBCsTB(copy)",bdyRange,
			   KOKKOS_LAMBDA (const idx_t i,const idx_t j,const idx_t k) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       tDstSub(i,j,k,nV) = bSrcSub(i,j,k,nV);
			       bDstSub(i,j,k,nV) = tSrcSub(i,j,k,nV);
			     }
			   });
      return;
    }
    
    // Pack and send bottom src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0) {
      Kokkos::parallel_for("Solver::commCellBCsTB(pack bottom)",bdyRange,
			   KOKKOS_LAMBDA (const idx_t i,const idx_t j,const idx_t k) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       bSrc(i,j,k,nV) = bSrcSub(i,j,k,nV);
			     }
			   });
      
      MPI_Send(bSrc.data(),int(bSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::bottom],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv top buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::top] >= 0) {
      MPI_Recv(tDst.data(),int(tDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::top],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsTB(pack/unpack top)",bdyRange,
			   KOKKOS_LAMBDA (const idx_t i,const idx_t j,const idx_t k) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       tDstSub(i,j,k,nV) = tDst(i,j,k,nV);
			       tSrc(i,j,k,nV) = tSrcSub(i,j,k,nV);
			     }
			   });
      
      MPI_Send(tSrc.data(),int(tSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::top],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack bottom buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0) {
      MPI_Recv(bDst.data(),int(bDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::bottom],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commCellBCsTB(unpack bottom)",bdyRange,
			   KOKKOS_LAMBDA (const idx_t i,const idx_t j,const idx_t k) {
			     for (idx_t nV=0; nV<NUM_VARS; nV++) {
			       bDstSub(i,j,k,nV) = bDst(i,j,k,nV);
			     }
			   });
    }
  }

  void Solver::commFaceBCsTB()
  {
    // Overall order: send bottom, recv top, send top, recv bottom

    // Range for (un)packing buffers
    auto bdyRange = Kokkos::MDRangePolicy<ExecSpace,Kokkos::Rank<SPACE_DIM - 1>,Kokkos::IndexType<idx_t>>
      ({0,0},{ps.nX,ps.nY});

    // Subviews of faceVals to pack from/unpack to
    auto bSrcSub = Kokkos::subview(faceVals.zDir,
				   Kokkos::ALL,Kokkos::ALL,0,
				   1,Kokkos::ALL,Kokkos::ALL);
    auto bDstSub = Kokkos::subview(faceVals.zDir,
				   Kokkos::ALL,Kokkos::ALL,0,
				   0,Kokkos::ALL,Kokkos::ALL);
    auto tSrcSub = Kokkos::subview(faceVals.zDir,
				   Kokkos::ALL,Kokkos::ALL,ps.nZ,
				   0,Kokkos::ALL,Kokkos::ALL);
    auto tDstSub = Kokkos::subview(faceVals.zDir,
				   Kokkos::ALL,Kokkos::ALL,ps.nZ,
				   1,Kokkos::ALL,Kokkos::ALL);

    // Communication buffers
    auto bSrc = bdyData.bFaceSrc;
    auto bDst = bdyData.bFaceDst;
    auto tSrc = bdyData.tFaceSrc;
    auto tDst = bdyData.tFaceDst;

    // Don't do any comms if sending to self (periodic BCs with one block)
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] == ps.layoutMPI.nbrRank[FaceLabel::top]) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(copy)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t j)) {
			     const idx_t nQuad = NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 tDstSub(i,j,nQ,nV) = bSrcSub(i,j,nQ,nV);
				 bDstSub(i,j,nQ,nV) = tSrcSub(i,j,nQ,nV);
			       }
			     }
			   });
      return;
    }
    
    // Pack and send bottom src buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0) {
      Kokkos::parallel_for("Solver::commFaceBCsTB(pack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t j)) {
			     const idx_t nQuad = NUM_QUAD_PTS*NUM_QUAD_PTS;
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 bSrc(i,j,nQ,nV) = bSrcSub(i,j,nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(bSrc.data(),int(bSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::bottom],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }

    // recv top buffer, unpack, pack, send
    if (ps.layoutMPI.nbrRank[FaceLabel::top] >= 0) {
      MPI_Recv(tDst.data(),int(tDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::top],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsTB(pack/unpack top)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t j)) {
			     const idx_t nQuad = NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 tDstSub(i,j,nQ,nV) = tDst(i,j,nQ,nV);
				 tSrc(i,j,nQ,nV) = tSrcSub(i,j,nQ,nV);
			       }
			     }
			   });
      
      MPI_Send(tSrc.data(),int(tSrc.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::top],KFVM_MPI_TAG,MPI_COMM_WORLD);
    }
    
    // recv and unpack bottom buffer
    if (ps.layoutMPI.nbrRank[FaceLabel::bottom] >= 0) {
      MPI_Recv(bDst.data(),int(bDst.size()),ps.layoutMPI.datatype,
	       ps.layoutMPI.nbrRank[FaceLabel::bottom],KFVM_MPI_TAG,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      Kokkos::parallel_for("Solver::commFaceBCsTB(unpack bottom)",bdyRange,
			   KOKKOS_LAMBDA (KFVM_DM_DECL(const idx_t i,const idx_t j)) {
			     const idx_t nQuad = NUM_QUAD_PTS*NUM_QUAD_PTS;			     
			     for (idx_t nQ=0; nQ<nQuad; nQ++) {
			       for (idx_t nV=0; nV<NUM_VARS; nV++) {
				 bDstSub(i,j,nQ,nV) = bDst(i,j,nQ,nV);
			       }
			     }
			   });
    }
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
      std::printf("Warning: Western cell BC undefined.\n");
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
      std::printf("Warning: Eastern cell BC undefined.\n");
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
      std::printf("Warning: Southern cell BC undefined.\n");
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
      std::printf("Warning: Northern cell BC undefined.\n");
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
      std::printf("Warning: Bottom cell BC undefined.\n");
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
      std::printf("Warning: Top cell BC undefined.\n");
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
      std::printf("Warning: Western face BC undefined.\n");
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
      std::printf("Warning: Eastern face BC undefined.\n");
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
      std::printf("Warning: Southern face BC undefined.\n");
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
      std::printf("Warning: Northern face BC undefined.\n");
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
      std::printf("Warning: Bottom face BC undefined.\n");
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
      std::printf("Warning: Top face BC undefined.\n");
    }
  }
#endif

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
