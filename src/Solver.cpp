// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

#include <Kokkos_Core.hpp>
#include <Kokkos_Macros.hpp>
#include <array>
#include <cstdint>
#include <cstdio>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include "Dimension.H"
#include "ProblemSetup.H"
#include "SimVar.H"
#include "SolutionWriter.H"
#include "Types.H"
#include "Numeric.H"
#include "Numeric_K.H"
#include "Hydro_K.H"

#include "Solver.H"

namespace KFVM {
  
  Solver::Solver(const ProblemSetup& a_ps):
    ps(a_ps),
    solWriter(a_ps),
    geom(ps),
    U_halo("U",KFVM_D_DECL(ps.nX+2*ps.rad,ps.nY+2*ps.rad,ps.nZ+2*ps.rad)),
    U1_halo("U_stage1",KFVM_D_DECL(ps.nX+2*ps.rad,ps.nY+2*ps.rad,ps.nZ+2*ps.rad)),
    U2_halo("U_stage2",KFVM_D_DECL(ps.nX+2*ps.rad,ps.nY+2*ps.rad,ps.nZ+2*ps.rad)),
    U3_halo("U_stage3",KFVM_D_DECL(ps.nX+2*ps.rad,ps.nY+2*ps.rad,ps.nZ+2*ps.rad)),
    U4_halo("U_stage4",KFVM_D_DECL(ps.nX+2*ps.rad,ps.nY+2*ps.rad,ps.nZ+2*ps.rad)),
    K("RHS",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    Ktil("RHS_til",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    FaceVals("FaceVals",KFVM_D_DECL(ps.nX+2,ps.nY+2,ps.nZ+2)),
    time(ps.initialTime),
    dt(ps.initialDeltaT),
    lastTimeStep(false)
  {
    setIC();
    solWriter.write(U_halo,0,time);
  }

  // Solve system for full time range
  void Solver::Solve()
  {
    // Evolve in time, recording solutions as needed
    // Start at nT=1 since IC is step 0
    for (std::size_t nT=1; nT<ps.maxTimeSteps && !lastTimeStep; ++nT) {
      std::printf("Step %ld: ",nT);
      TakeStep();
      if (nT%ps.plotFreq == 0 || lastTimeStep || nT==ps.maxTimeSteps-1) {
        solWriter.write(U_halo,nT,time);
      }
    }
  }

  void Solver::TakeStep()
  {
    // Grab reference to RK coefficients
    Numeric::SSPCoeff<5> rk;

    // Set range policy for summing stages together
    auto rangePolicy =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},
						     {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    
    // Evaluate RHS and set dt
    Real maxVel = evalRHS(U_halo,K,time);
    std::printf("MaxVel = %e\n",maxVel);
    dt = std::fmin(ps.cfl*std::fmin(ps.dx,ps.dy)/maxVel,2*dt);
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
    Kokkos::parallel_for("RKStage_1",rangePolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     U1(KFVM_D_DECL(i,j,k),nV) =
			       rk.a_1_0*U(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_1*K(KFVM_D_DECL(i,j,k),nV);
			   }
			 });
    solWriter.write(U1_halo,-1,time);

    // Second stage
    evalRHS(U1_halo,K,time + dt/2.0);
    auto U2 = trimCellHalo(U2_halo);
    Kokkos::parallel_for("RKStage_2",rangePolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     U2(KFVM_D_DECL(i,j,k),nV) =
			       rk.a_2_0*U(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_2_1*U1(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_2*K(KFVM_D_DECL(i,j,k),nV);
			   }
			 });
    solWriter.write(U2_halo,-2,time);

    // Third stage
    evalRHS(U2_halo,K,time + dt/2.0);
    auto U3 = trimCellHalo(U3_halo);
    Kokkos::parallel_for("RKStage_3",rangePolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     U3(KFVM_D_DECL(i,j,k),nV) =
			       rk.a_3_0*U(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_3_2*U2(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_3*K(KFVM_D_DECL(i,j,k),nV);
			   }
			 });
    solWriter.write(U3_halo,-3,time);

    // Fourth stage
    evalRHS(U3_halo,K,time + dt/2.0);
    auto U4 = trimCellHalo(U4_halo);
    Kokkos::parallel_for("RKStage_4",rangePolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     U4(KFVM_D_DECL(i,j,k),nV) =
			       rk.a_4_0*U(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_4_3*U3(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_4*K(KFVM_D_DECL(i,j,k),nV);
			   }
			 });
    solWriter.write(U4_halo,-4,time);

    // Fifth stage
    evalRHS(U4_halo,Ktil,time + dt/2.0);
    Kokkos::parallel_for("RKStage_5",rangePolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     U(KFVM_D_DECL(i,j,k),nV) =
			       rk.a_5_0*U(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_5_2*U2(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_5_3*U3(KFVM_D_DECL(i,j,k),nV) +
			       rk.a_5_4*U4(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_5*K(KFVM_D_DECL(i,j,k),nV) +
			       dt*rk.b_5til*Ktil(KFVM_D_DECL(i,j,k),nV);
			   }
			 });

    time += dt;
  }

  Real Solver::evalRHS(CellDataView sol_halo,CellDataView rhs,Real t)
  {
    // Set BCs on cell averages
    setCellBCs(sol_halo,t);

    // Reconstruct face states
    // Temporary: hard coded to first order Godunov
    auto U = trimCellHalo(sol_halo);
    auto RS = trimFaceHalo(FaceVals);
    auto cellRngPolicy =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},
						     {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_for("FaceRecon",cellRngPolicy,
			 KOKKOS_LAMBDA (KFVM_D_DECL(const int i,const int j,const int k)) {
			   // Loop over quadrature points
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
#if (SPACE_DIM == 3)
			     for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
#endif
			       // Loop over SimVars
			       for (int nV=0; nV<NUM_VARS; nV++) {
#if (SPACE_DIM == 2)
				 RS(i,j,FaceLabel::west,nQ,nV) = U(i,j,nV);
				 RS(i,j,FaceLabel::east,nQ,nV) = U(i,j,nV);
				 RS(i,j,FaceLabel::south,nQ,nV) = U(i,j,nV);
				 RS(i,j,FaceLabel::north,nQ,nV) = U(i,j,nV);
#else
				 RS(i,j,k,FaceLabel::west,nQ,nR,nV) = U(i,j,k,nV);
				 RS(i,j,k,FaceLabel::east,nQ,nR,nV) = U(i,j,k,nV);
				 RS(i,j,k,FaceLabel::south,nQ,nR,nV) = U(i,j,k,nV);
				 RS(i,j,k,FaceLabel::north,nQ,nR,nV) = U(i,j,k,nV);
				 RS(i,j,k,FaceLabel::bottom,nQ,nR,nV) = U(i,j,k,nV);
				 RS(i,j,k,FaceLabel::top,nQ,nR,nV) = U(i,j,k,nV);
#endif
			       }
#if (SPACE_DIM == 3)
			     }
#endif
			   }
			 });

    // Set BCs on Riemann states
    setFaceBCs(t);
    
    // Call Riemann solver
    auto fluxRngPolicy_bulk =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>,Hydro::RiemannSolver_K<decltype(RS)>::BulkTag>({KFVM_D_DECL(0,0,0)},
												   {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    auto fluxRngPolicy_east = Kokkos::RangePolicy<Hydro::RiemannSolver_K<decltype(RS)>::EastTag>({0,ps.nY});
    auto fluxRngPolicy_north = Kokkos::RangePolicy<Hydro::RiemannSolver_K<decltype(RS)>::NorthTag>({0,ps.nX});
    Real vBulk,vEast,vNorth = 0.0;
    Kokkos::parallel_reduce("RiemannSolver::Bulk",fluxRngPolicy_bulk,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vBulk));
    Kokkos::parallel_reduce("RiemannSolver::East",fluxRngPolicy_east,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vEast));
    Kokkos::parallel_reduce("RiemannSolver::North",fluxRngPolicy_north,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vNorth));
    Real maxVel = std::fmax(vBulk,std::fmax(vEast,vNorth));
#if (SPACE_DIM == 3)    
    Kokkos::parallel_reduce("RiemannSolver::Top",fluxRngPolicy_top,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(maxVel));
#endif

    // Integrate fluxes and store into rhs
    Numeric::QuadRule<NUM_QUAD_PTS> qr;
    Kokkos::parallel_for("IntegrateFlux",cellRngPolicy,
			 KOKKOS_LAMBDA (const int i,const int j) {
			   // Zero out RHS before accumulating fluxes
			   for (int nV=0; nV<NUM_VARS; nV++) {
			     rhs(i,j,nV) = 0.0;
			   }
			   
			   // Loop over quadrature points and components
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
			     for (int nV=0; nV<NUM_VARS; nV++) {
			       // x-direction
			       rhs(i,j,nV) += qr.wt(nQ)*(RS(i,j,FaceLabel::west,nQ,nV) - RS(i,j,FaceLabel::east,nQ,nV))/ps.dx;
			       // y-direction
			       rhs(i,j,nV) += qr.wt(nQ)*(RS(i,j,FaceLabel::south,nQ,nV) - RS(i,j,FaceLabel::north,nQ,nV))/ps.dy;
			     }
			   }
			 });
    
    return maxVel;
  }

void Solver::setCellBCs(CellDataView sol_halo,Real t)
  {
    (void) sol_halo;
    (void) t;
  }

  // Hardcoded to 2D outflow for now
  void Solver::setFaceBCs(Real t)
  {
    (void) t;
    auto faceRngPolicy_ew = Kokkos::RangePolicy<>({1,ps.nY+1});
    auto faceRngPolicy_ns = Kokkos::RangePolicy<>({1,ps.nX+1});

    Kokkos::parallel_for("FaceBCs::East",faceRngPolicy_ew,
			 KOKKOS_LAMBDA (const int j) {
			   // Loop over quadrature points
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
			     // Loop over SimVars
			       for (int nV=0; nV<NUM_VARS; nV++) {
				 FaceVals(ps.nX+1,j,FaceLabel::west,nQ,nV) = FaceVals(ps.nX,j,FaceLabel::east,nQ,nV);
			       }
			   }
			 });

    Kokkos::parallel_for("FaceBCs::West",faceRngPolicy_ew,
			 KOKKOS_LAMBDA (const int j) {
			   // Loop over quadrature points
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
			     // Loop over SimVars
			       for (int nV=0; nV<NUM_VARS; nV++) {
				 FaceVals(0,j,FaceLabel::east,nQ,nV) = FaceVals(1,j,FaceLabel::west,nQ,nV);
			       }
			   }
			 });

    Kokkos::parallel_for("FaceBCs::North",faceRngPolicy_ns,
			 KOKKOS_LAMBDA (const int i) {
			   // Loop over quadrature points
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
			     // Loop over SimVars
			       for (int nV=0; nV<NUM_VARS; nV++) {
				 FaceVals(i,ps.nY+1,FaceLabel::south,nQ,nV) = FaceVals(i,ps.nY,FaceLabel::north,nQ,nV);
			       }
			   }
			 });

    Kokkos::parallel_for("FaceBCs::South",faceRngPolicy_ns,
			 KOKKOS_LAMBDA (const int i) {
			   // Loop over quadrature points
			   for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
			     // Loop over SimVars
			       for (int nV=0; nV<NUM_VARS; nV++) {
				 FaceVals(i,0,FaceLabel::north,nQ,nV) = FaceVals(i,1,FaceLabel::south,nQ,nV);
			       }
			   }
			 });
  }

  void Solver::setIC()
  {
    Numeric::QuadRule<NUM_QUAD_PTS> qr;
    auto U = trimCellHalo(U_halo);
    Kokkos::parallel_for("IntegrateIC",
			 Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)}),
			 Numeric::IntegrateIC_K<decltype(U),decltype(qr.ab),NUM_QUAD_PTS>(U,qr.ab,qr.wt,geom));
  }
  
} // end namespace KFVM
