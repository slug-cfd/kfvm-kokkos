#include <cstdio>
#include <typeinfo>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "ProblemSetup.H"
#include "Solver.H"

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    printf("Kokkos initialized with default execution space: %s\n",
	   typeid(Kokkos::DefaultExecutionSpace).name());
    
    // Create ProblemSetup object with all settings and configuration
    // KFVM::ProblemSetup ps("data/SodX",
    //     		  256,8,8,
    //     		  -1.0,1.0,0.0,8.0/128.0,0.0,8.0/128.0,
    //     		  0.2,20,
    //     		  1.4);

    // KFVM::ProblemSetup ps("data/SodY",
    // 			  8,256,8,
    // 			  0.0,8.0/128.0,-1.0,1.0,0.0,8.0/128.0,
    // 			  0.2,20,
    // 			  1.4);

    // KFVM::ProblemSetup ps("data/SodZ",
    // 			  8,8,256,
    // 			  0.0,8.0/128.0,0.0,8.0/128.0,-1.0,1.0,
    // 			  0.2,20,
    // 			  1.4);
    
    // KFVM::ProblemSetup ps("data/Riem_2d_3c",
    //     		  256,256,8,
    //     		  0.0,1.0,0.0,1.0,0.0,8.0/256.0,
    //     		  0.8,100,
    //     		  1.4);
    
    // KFVM::ProblemSetup ps("data/Riem_3d_1c",
    // 			  96,96,96,
    // 			  -1.0,1.0,-1.0,1.0,-1.0,1.0,
    // 			  0.53,100,
    // 			  1.4);
    
    // KFVM::ProblemSetup ps("data/Implosion",
    // 			  64,64,64,
    // 			  0.0,0.3,0.0,0.3,0.0,0.3,
    // 			  2.5,100,
    // 			  1.4);
    
    // KFVM::ProblemSetup ps("data/Implosion",
    // 			  128,96,64,
    // 			  0.0,0.3,0.0,0.225,0.0,0.15,
    // 			  2.5,40,
    // 			  1.4);
    
    KFVM::ProblemSetup ps("data/IsenVort",
        		  128,128,8,
        		  -10.0,10.0,-10.0,10.0,0.0,160.0/128.0,
        		  20.0,20,
        		  1.4);
    
    // KFVM::ProblemSetup ps("data/LinAdv",
    //     		  128,128,8,
    //     		  -10.0,10.0,-10.0,10.0,0.0,160.0/128.0,
    //     		  20.0,5,
    //     		  1.4);
    
    // KFVM::ProblemSetup ps("data/BuchHelz",
    //     		  128,128,8,
    //                       -1.0,1.0,-1.0,1.0,0.0,16.0/128.0,
    //     		  1.0,20,
    //     		  1.4);
    
    // KFVM::ProblemSetup ps("data/DoubleMach",
    // 			  256,64,16,
    // 			  0.0,4.0,0.0,1.0,0.0,0.25,
    // 			  0.25,5,
    // 			  1.4);
    
    // KFVM::ProblemSetup ps("data/TaylorGreen",
    // 			  96,96,96,
    // 			  0.0,2.0*M_PI,0.0,2.0*M_PI,0.0,2.0*M_PI,
    // 			  10.75,100,
    // 			  1.4);

    // Create solver object to store and advance solution
    KFVM::Solver solver(ps);

    // Solve until final time
    solver.Solve();
    //solver.TakeStep();
  }
  Kokkos::finalize();
}
