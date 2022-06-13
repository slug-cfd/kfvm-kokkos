// File: Solver.cpp
// Purpose: The solver class is responsible for holding the
//          solution and evolving it through time

#include <KokkosExp_MDRangePolicy.hpp>
#include <Kokkos_Rank.hpp>
#include <array>
#include <impl/Kokkos_HostThreadTeam.hpp>

#include "Dimension.H"
#include "ProblemSetup.H"
#include "numeric/Numeric.H"
#include "numeric/Numeric_K.H"

#include "Solver.H"

namespace KFVM {
  Solver::Solver(const ProblemSetup& a_ps):
    ps(a_ps),
    geom(ps),
    U("U",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U1("U_stage1",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U2("U_stage2",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U3("U_stage3",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    U4("U_stage4",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    K("RHS",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
    Ktil("RHS_til",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ))
  {
    // Set initial condition
    Numeric::QuadRule<NUM_QUAD_PTS> qr;
    Kokkos::parallel_for("IntegrateIC",
			 Kokkos::MDRangePolicy<Kokkos::Rank<SPACE_DIM>>({KFVM_D_DECL(0,0,0)},{KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)}),
			 Numeric::IntegrateIC_K<decltype(U),decltype(qr.ab),NUM_QUAD_PTS>(U,qr.ab,qr.wt,geom));
  }
}
