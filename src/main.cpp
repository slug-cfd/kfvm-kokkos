#include <cstdio>
#include <string>
#include <typeinfo>

#include <fmt/color.h>
#include <fmt/core.h>

#include <mpi.h>

#include <Kokkos_Core.hpp>

#include <pdi.h>

#include <Definitions.H>

#include "PrinterMPI.H"
#include "ProblemSetup.H"
#include "Solver.H"
#include "Types.H"
#include "physics/EquationTypes.H"
#include "physics/Physics_K.H"

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  {
    Kokkos::initialize(argc, argv);
    {
      if (argc > 1) {
        // Create Problemsetup object
        KFVM::ProblemSetup ps;
        if (ps.setFromFile(argv[1])) {
          ps.print();
        }

        // Initialize PDI reading YAML file
        PC_tree_t pdi_conf = PC_parse_path(ps.pdiConf.c_str());
        PDI_init(PC_get(pdi_conf, ".pdi"));

        // Create solver object and run to final time
        KFVM::Solver solver(ps);
        solver.Solve();

        // Finalize PDI and clean up
        PDI_finalize();
        PC_tree_destroy(&pdi_conf);
      } else {
        fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                   "[Error] Input file must be provided. Exiting now\n");
      }
    }
    Kokkos::finalize();
  }
  MPI_Finalize();
  return 0;
}
