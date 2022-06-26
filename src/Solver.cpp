// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

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

#include "Dimension.H"
#include "ProblemSetup.H"
#include "SimVar.H"
#include "Types.H"
#include "BoundaryConditions_K.H"
#include "NetCDFWriter.H"
#include "numeric/Numeric.H"
#include "numeric/Numeric_K.H"
#include "hydro/Hydro_K.H"

#include "Solver.H"
#include "stencil/Stencil_K.H"

namespace KFVM {
  
  Solver::Solver(const ProblemSetup& a_ps):
    ps(a_ps),
    netCDFWriter(a_ps),
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
    netCDFWriter.write(U_halo,0,time);
  }

  // Solve system for full time range
  void Solver::Solve()
  {
    Kokkos::Profiling::pushRegion("Solver::Solve");
    
    // Evolve in time, recording solutions as needed
    // Start at nT=1 since IC is step 0
    for (std::size_t nT=1; nT<ps.maxTimeSteps && !lastTimeStep; ++nT) {
      std::printf("Step %ld: ",nT);
      TakeStep();
      if (nT%ps.plotFreq == 0 || lastTimeStep || nT==ps.maxTimeSteps-1) {
        netCDFWriter.write(U_halo,nT,time);
      }
    }

    Kokkos::Profiling::popRegion();
  }

  void Solver::TakeStep()
  {
    Kokkos::Profiling::pushRegion("Solver::TakeStep");

    // Set range policy for summing stages together
    auto rangePolicy =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},
						     {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    
    // Evaluate RHS and set dt
    Real maxVel = evalRHS(U_halo,K,time);
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
			 Numeric::SSP45_S1_K<decltype(U),decltype(K),5>(U1,U,K,Numeric::SSPCoeff<5>(),dt));

    // Second stage
    evalRHS(U1_halo,K,time + dt/2.0);
    auto U2 = trimCellHalo(U2_halo);
    Kokkos::parallel_for("RKStage_2",rangePolicy,
			 Numeric::SSP45_S2_K<decltype(U),decltype(K),5>(U2,U,U1,K,Numeric::SSPCoeff<5>(),dt));

    // Third stage
    evalRHS(U2_halo,K,time + dt/2.0);
    auto U3 = trimCellHalo(U3_halo);
    Kokkos::parallel_for("RKStage_3",rangePolicy,
			 Numeric::SSP45_S3_K<decltype(U),decltype(K),5>(U3,U,U2,K,Numeric::SSPCoeff<5>(),dt));

    // Fourth stage
    evalRHS(U3_halo,K,time + dt/2.0);
    auto U4 = trimCellHalo(U4_halo);
    Kokkos::parallel_for("RKStage_4",rangePolicy,
			 Numeric::SSP45_S4_K<decltype(U),decltype(K),5>(U4,U,U3,K,Numeric::SSPCoeff<5>(),dt));

    // Fifth stage
    evalRHS(U4_halo,Ktil,time + dt/2.0);
    Kokkos::parallel_for("RKStage_5",rangePolicy,
			 Numeric::SSP45_S5_K<decltype(U),decltype(K),5>(U,U2,U3,U4,K,Ktil,Numeric::SSPCoeff<5>(),dt));

    time += dt;

    Kokkos::Profiling::popRegion();
  }

  Real Solver::evalRHS(CellDataView sol_halo,CellDataView rhs,Real t)
  {
    Kokkos::Profiling::pushRegion("Solver::evalRHS");
    
    // Set BCs on cell averages
    setCellBCs(sol_halo,t);

    // Reconstruct face states
    // Temporary: hard coded to first order Godunov
    auto U = trimCellHalo(sol_halo);
    auto RS = trimFaceHalo(FaceVals);
    auto cellRngPolicy =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},
						     {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_for("FaceRecon",cellRngPolicy,Stencil::MinModRecon_K<decltype(U),decltype(RS)>(U,RS));

    // Set BCs on Riemann states
    setFaceBCs(t);
    
    // Call Riemann solver
    Real vBulk = 0.0,vEast = 0.0,vNorth = 0.0,vTop = 0.0;
    auto fluxRngPolicy_bulk =
      Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>,Hydro::RiemannSolver_K<decltype(RS)>::BulkTag>({KFVM_D_DECL(0,0,0)},
												   {KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)});
    Kokkos::parallel_reduce("RiemannSolver::Bulk",fluxRngPolicy_bulk,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vBulk));
#if (SPACE_DIM == 2)
    auto fluxRngPolicy_east = Kokkos::RangePolicy<Hydro::RiemannSolver_K<decltype(RS)>::EastTag>({0,ps.nY});
    auto fluxRngPolicy_north = Kokkos::RangePolicy<Hydro::RiemannSolver_K<decltype(RS)>::NorthTag>({0,ps.nX});
#else
    auto fluxRngPolicy_east =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>,Hydro::RiemannSolver_K<decltype(RS)>::EastTag>({0,0},{ps.nY,ps.nZ});
    auto fluxRngPolicy_north =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>,Hydro::RiemannSolver_K<decltype(RS)>::NorthTag>({0,0},{ps.nX,ps.nZ});
    auto fluxRngPolicy_top =
      Kokkos::MDRangePolicy<Kokkos::Rank<2>,Hydro::RiemannSolver_K<decltype(RS)>::TopTag>({0,0},{ps.nX,ps.nY});
    Kokkos::parallel_reduce("RiemannSolver::Top",fluxRngPolicy_top,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vTop));
#endif
    Kokkos::parallel_reduce("RiemannSolver::East",fluxRngPolicy_east,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vEast));
    Kokkos::parallel_reduce("RiemannSolver::North",fluxRngPolicy_north,
			    Hydro::RiemannSolver_K<decltype(RS)>(RS,ps),
			    Kokkos::Max<Real>(vNorth));
    Real maxVel = std::fmax(vBulk,std::fmax(vEast,std::fmax(vNorth,vTop)));

    // Integrate fluxes and store into rhs
    Kokkos::parallel_for("IntegrateFlux",cellRngPolicy,
			 Numeric::IntegrateFlux_K<decltype(rhs),decltype(RS),decltype(qr.ab),NUM_QUAD_PTS>(rhs,RS,qr.ab,qr.wt,geom));
    
    Kokkos::Profiling::popRegion();
    return maxVel;
  }

void Solver::setCellBCs(CellDataView sol_halo,Real t)
  {
    (void) t;
    Kokkos::Profiling::pushRegion("Solver::setCellBCs");
#if (SPACE_DIM == 2)
    auto rngPolicy_ew = Kokkos::RangePolicy<>({ps.rad,ps.rad + ps.nY  });
    auto rngPolicy_ns = Kokkos::RangePolicy<>({0     ,ps.nX + 2*ps.rad});
#else
    auto rngPolicy_ew = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({int(ps.rad),int(ps.rad)},{int(ps.nY + ps.rad)  ,int(ps.nZ + ps.rad)  });
    auto rngPolicy_ns = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0     ,int(ps.rad)},{int(ps.nX + 2*ps.rad),int(ps.nZ + ps.rad)  });
    auto rngPolicy_tb = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0     ,0     },{int(ps.nX + 2*ps.rad),int(ps.nY + 2*ps.rad)});
#endif
    Kokkos::parallel_for("CellBCs::EastWest",rngPolicy_ew,
			 CellBcEW_K<decltype(sol_halo)>(sol_halo,ps.rad,ps.nX));
    
    Kokkos::parallel_for("CellBCs::NorthSouth",rngPolicy_ns,
			 CellBcNS_K<decltype(sol_halo)>(sol_halo,ps.rad,ps.nY));
#if (SPACE_DIM == 3)
    Kokkos::parallel_for("CellBCs::TopBottom",rngPolicy_tb,
			 CellBcTB_K<decltype(sol_halo)>(sol_halo,ps.rad,ps.nZ));
#endif
    Kokkos::Profiling::popRegion();
  }

  // Hardcoded to outflow for now
  void Solver::setFaceBCs(Real t)
  {
    (void) t;
    Kokkos::Profiling::pushRegion("Solver::setFaceBCs");
#if (SPACE_DIM == 2)
    auto faceRngPolicy_ew = Kokkos::RangePolicy<>({1,ps.nY + 1});
    auto faceRngPolicy_ns = Kokkos::RangePolicy<>({1,ps.nX + 1});
#else
    auto faceRngPolicy_ew = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{ps.nY + 1,ps.nZ + 1});
    auto faceRngPolicy_ns = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{ps.nX + 1,ps.nZ + 1});
    auto faceRngPolicy_tb = Kokkos::MDRangePolicy<Kokkos::Rank<2>>({1,1},{ps.nX + 1,ps.nY + 1});
#endif

    Kokkos::parallel_for("FaceBCs::EastWest",faceRngPolicy_ew,
			 FaceBcEW_K<decltype(FaceVals)>(FaceVals,ps.rad,ps.nX));

    Kokkos::parallel_for("FaceBCs::NorthSouth",faceRngPolicy_ns,
			 FaceBcNS_K<decltype(FaceVals)>(FaceVals,ps.rad,ps.nY));
#if (SPACE_DIM == 3)
    Kokkos::parallel_for("FaceBCs::TopBottom",faceRngPolicy_tb,
			 FaceBcTB_K<decltype(FaceVals)>(FaceVals,ps.rad,ps.nZ));
#endif
    Kokkos::Profiling::popRegion();
  }

  void Solver::setIC()
  {
    Kokkos::Profiling::pushRegion("Solver::setIC");
    
    auto U = trimCellHalo(U_halo);
    Kokkos::parallel_for("IntegrateIC",
			 Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)}),
			 Numeric::IntegrateIC_K<decltype(U),decltype(qr.ab),NUM_QUAD_PTS>(U,qr.ab,qr.wt,geom));

    Kokkos::Profiling::popRegion();
  }
  
} // end namespace KFVM
