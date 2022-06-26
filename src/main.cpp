#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

#include "ProblemSetup.H"
#include "Solver.H"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    printf("Kokkos initialized with default execution space: %s\n",typeid(Kokkos::DefaultExecutionSpace).name());
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/Sod",
    // 			  128,16,16,
    // 			  0.0,1.0,0.0,0.125,0.0,0.125,
    // 			  0.2,
    // 			  1.4);
    
    // Create ProblemSetup object with all settings and configuration
    KFVM::ProblemSetup ps("data/Riem_2d_3c",
			  1024,1024,8,
			  0.0,1.0,0.0,1.0,0.0,8.0/1024.0,
			  0.8,
			  1.4);
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/Riem_3d_1c",
    // 			  64,64,64,//128,128,128,
    // 			  -1.0,1.0,-1.0,1.0,-1.0,1.0,
    // 			  0.53,
    // 			  1.4);

    // Create solver object to store and advance solution
    KFVM::Solver solver(ps);

    // Take one step with the solver for testing
    //solver.TakeStep();
    solver.Solve();
  }
  Kokkos::finalize();
}
