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
      KFVM::Real vx = 0.5,vy = 0.0,vz = 0.0,W = 1.0/Kokkos::sqrt(1.0 - (vx*vx + vy*vy + vz*vz));
      KFVM::Physics::SRHydro::PrimVec prim = {1.0,vx,vy,vz,W,1.0};
      KFVM::Physics::SRHydro::ConsVec cons;
      KFVM::Physics::EOS<KFVM::EquationType::SRHydro>::prim2cons(4.0/3.0,prim,cons);
      KFVM::Physics::VarConvert<KFVM::EquationType::SRHydro,CallVec> varConv(cons,4.0/3.0);
      std::printf("\nJ = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]]); ",
		  varConv.J[0][0],varConv.J[0][1],varConv.J[0][2],varConv.J[0][3],varConv.J[0][4],
		  varConv.J[1][0],varConv.J[1][1],varConv.J[1][2],varConv.J[1][3],varConv.J[1][4],
		  varConv.J[2][0],varConv.J[2][1],varConv.J[2][2],varConv.J[2][3],varConv.J[2][4],
		  varConv.J[3][0],varConv.J[3][1],varConv.J[3][2],varConv.J[3][3],varConv.J[3][4],
		  varConv.J[4][0],varConv.J[4][1],varConv.J[4][2],varConv.J[4][3],varConv.J[4][4]);
      std::printf("Jinv = np.array([[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e],[%e,%e,%e,%e,%e]]); J @ Jinv\n",
		  varConv.Jinv[0][0],varConv.Jinv[0][1],varConv.Jinv[0][2],varConv.Jinv[0][3],varConv.Jinv[0][4],
		  varConv.Jinv[1][0],varConv.Jinv[1][1],varConv.Jinv[1][2],varConv.Jinv[1][3],varConv.Jinv[1][4],
		  varConv.Jinv[2][0],varConv.Jinv[2][1],varConv.Jinv[2][2],varConv.Jinv[2][3],varConv.Jinv[2][4],
		  varConv.Jinv[3][0],varConv.Jinv[3][1],varConv.Jinv[3][2],varConv.Jinv[3][3],varConv.Jinv[3][4],
		  varConv.Jinv[4][0],varConv.Jinv[4][1],varConv.Jinv[4][2],varConv.Jinv[4][3],varConv.Jinv[4][4]);

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
