#include <cstdio>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"
#include "Solver.H"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    std::printf("Kokkos initialized with default execution space: %s\n",
		typeid(KFVM::ExecSpace).name());
    
    // Create ProblemSetup object with all settings and configuration
    KFVM::ProblemSetup ps;
    if (argc > 1) {
      if (ps.setFromFile(argv[1])) {
	ps.print();
	// Create solver object to store and advance solution
	KFVM::Solver solver(ps);
	solver.Solve();
      }
    } else {
      std::printf("Warning!!!! Input file must be provided. Exiting now\n");
    }
  }
  Kokkos::finalize();
}
