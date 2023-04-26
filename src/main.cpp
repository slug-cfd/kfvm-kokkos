#include <cstdio>
#include <typeinfo>
#include <string>

#include <Kokkos_Core.hpp>

#include <pdi.h>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"
#include "Solver.H"
#include "physics/EquationTypes.H"
#include "physics/Physics_K.H"
#include "physics/srhydro/SRHydro_SimVar.H"

struct CallVec {
  KFVM::Physics::SRHydro::ConsVec cons;
  CallVec(KFVM::Physics::SRHydro::ConsVec cons_)
  {
    cons[0] = cons_[0];
    cons[1] = cons_[1];
    cons[2] = cons_[2];
    cons[3] = cons_[3];
    cons[4] = cons_[4];
  }
  KFVM::Real operator()(int n)
  {
    return cons[n];
  }
};    

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Kokkos initialized with default execution space: %s\n",
		typeid(KFVM::ExecSpace).name());

    if (argc > 1) {
      // Create Problemsetup object
      KFVM::ProblemSetup ps;
      if (ps.setFromFile(argv[1])) {
	ps.print();
      }

      // debug
      // KFVM::Real vx = 0.1,vy = 0.2,vz = -0.15,W = 1.0/Kokkos::sqrt(1.0 - (vx*vx + vy*vy + vz*vz));
      // std::printf("W = %lf\n",W);
      // KFVM::Physics::SRHydro::PrimVec prim = {1.0,vx,vy,vz,W,1.0};
      // KFVM::Physics::SRHydro::ConsVec cons;
      // KFVM::Physics::EOS<KFVM::EquationType::SRHydro>::prim2cons(4.0/3.0,prim,cons);
      // KFVM::Physics::VarConvert<KFVM::EquationType::SRHydro,CallVec> varConv(cons,4.0/3.0);
      // std::printf("\np2c = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]]); ",
      // 		  varConv.p2c[0][0],varConv.p2c[0][1],varConv.p2c[0][2],varConv.p2c[0][3],varConv.p2c[0][4],
      // 		  varConv.p2c[1][0],varConv.p2c[1][1],varConv.p2c[1][2],varConv.p2c[1][3],varConv.p2c[1][4],
      // 		  varConv.p2c[2][0],varConv.p2c[2][1],varConv.p2c[2][2],varConv.p2c[2][3],varConv.p2c[2][4],
      // 		  varConv.p2c[3][0],varConv.p2c[3][1],varConv.p2c[3][2],varConv.p2c[3][3],varConv.p2c[3][4],
      // 		  varConv.p2c[4][0],varConv.p2c[4][1],varConv.p2c[4][2],varConv.p2c[4][3],varConv.p2c[4][4]);
      // std::printf("c2p = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]]); ",
      // 		  varConv.c2p[0][0],varConv.c2p[0][1],varConv.c2p[0][2],varConv.c2p[0][3],varConv.c2p[0][4],
      // 		  varConv.c2p[1][0],varConv.c2p[1][1],varConv.c2p[1][2],varConv.c2p[1][3],varConv.c2p[1][4],
      // 		  varConv.c2p[2][0],varConv.c2p[2][1],varConv.c2p[2][2],varConv.c2p[2][3],varConv.c2p[2][4],
      // 		  varConv.c2p[3][0],varConv.c2p[3][1],varConv.c2p[3][2],varConv.c2p[3][3],varConv.c2p[3][4],
      // 		  varConv.c2p[4][0],varConv.c2p[4][1],varConv.c2p[4][2],varConv.c2p[4][3],varConv.c2p[4][4]);
      // std::printf("p2ci = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]])\n",
      // 		  varConv.p2ci[0][0],varConv.p2ci[0][1],varConv.p2ci[0][2],varConv.p2ci[0][3],varConv.p2ci[0][4],
      // 		  varConv.p2ci[1][0],varConv.p2ci[1][1],varConv.p2ci[1][2],varConv.p2ci[1][3],varConv.p2ci[1][4],
      // 		  varConv.p2ci[2][0],varConv.p2ci[2][1],varConv.p2ci[2][2],varConv.p2ci[2][3],varConv.p2ci[2][4],
      // 		  varConv.p2ci[3][0],varConv.p2ci[3][1],varConv.p2ci[3][2],varConv.p2ci[3][3],varConv.p2ci[3][4],
      // 		  varConv.p2ci[4][0],varConv.p2ci[4][1],varConv.p2ci[4][2],varConv.p2ci[4][3],varConv.p2ci[4][4]);
      // std::printf("c2pi = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]])\n",
      // 		  varConv.c2pi[0][0],varConv.c2pi[0][1],varConv.c2pi[0][2],varConv.c2pi[0][3],varConv.c2pi[0][4],
      // 		  varConv.c2pi[1][0],varConv.c2pi[1][1],varConv.c2pi[1][2],varConv.c2pi[1][3],varConv.c2pi[1][4],
      // 		  varConv.c2pi[2][0],varConv.c2pi[2][1],varConv.c2pi[2][2],varConv.c2pi[2][3],varConv.c2pi[2][4],
      // 		  varConv.c2pi[3][0],varConv.c2pi[3][1],varConv.c2pi[3][2],varConv.c2pi[3][3],varConv.c2pi[3][4],
      // 		  varConv.c2pi[4][0],varConv.c2pi[4][1],varConv.c2pi[4][2],varConv.c2pi[4][3],varConv.c2pi[4][4]);

      // Initialize PDI reading YAML file
      PC_tree_t pdi_conf = PC_parse_path(ps.pdiConf.c_str());
      PDI_init(PC_get(pdi_conf,".pdi"));
    
      // Create solver object and run to final time
      KFVM::Solver solver(ps);
      solver.Solve();
      //solver.TakeStep();

      // Finalize PDI and clean up
      PDI_finalize();
      PC_tree_destroy(&pdi_conf);
    } else {
      std::printf("Error!!!! Input file must be provided. Exiting now\n");
    }
  }
  Kokkos::finalize();
}
