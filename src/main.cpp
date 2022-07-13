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
			  128,128,8,
			  0.0,1.0,0.0,1.0,0.0,8.0/128.0,
			  0.8,
			  1.4);
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/Riem_3d_1c",
    // 			  64,64,64,
    // 			  -1.0,1.0,-1.0,1.0,-1.0,1.0,
    // 			  0.53,
    // 			  1.4);
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/Implosion",
    // 			  128,128,128,
    // 			  0.0,0.3,0.0,0.3,0.0,0.3,
    // 			  2.5,
    // 			  1.4);
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/IsenVort",
    // 			  128,128,8,
    // 			  -10.0,10.0,-10.0,10.0,0.0,160.0/128.0,
    // 			  20.0,
    // 			  1.4);

    // Create solver object to store and advance solution
    KFVM::Solver solver(ps);

    // Solve until final time
    solver.Solve();
  }
  Kokkos::finalize();
}
