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
    KFVM::ProblemSetup ps("data/Sod",
			  128,16,1,
			  16,16,1,
			  0.0,1.0,0.0,0.125,0.0,1.0/128.0,
			  1.4);

    // Create solver object to store and advance solution
    KFVM::Solver solver(ps);
  }
  Kokkos::finalize();
}
