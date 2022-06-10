#ifndef NUMERIC_K_H_
#define NUMERIC_K_H_

#include <cstring>

#include <CL/sycl.hpp>

#include "Types.H"
#include "ProblemSetup.H"
#include "Numeric.H"

namespace KFVM {

  // Anonymous helpers
  namespace {

    inline Real sclAb(const Real ctr,const Real del,const Real ab)
    {
      return Real(0.5)*ab*del + ctr;
    }
    
  }

  // User defined functions
  namespace {
#include "InitialCondition.H"
  }

  namespace Numeric {

    void IntegrateIC_K(const ProblemSetup& ps,
                       cl::sycl::queue& d_queue,
                       cl::sycl::range<SPACE_DIM>& cellTile_rg,
                       cl::sycl::buffer<Real,SPACE_DIM>& U)
    {
      d_queue.submit([&](cl::sycl::handler& cgh) {
        cl::sycl::accessor acc {U,cgh,cl::sycl::write_only,cl::sycl::no_init};
        // Pull out data needed by kernel to avoid capture by ref/value confusion
        // Is there a better way?
        std::size_t rad = ps.rad;
        Real dx = ps.dx,dy = ps.dy;
        Real xLo = ps.xLo,yLo = ps.yLo;
        FluidProperties fProp = ps.fluidProp;
        cgh.parallel_for<class EvalInitialCondition> (
          cellTile_rg,[=](cl::sycl::id<SPACE_DIM> idx){
            auto gId = idx + cl::sycl::id(rad,rad);
            auto dst = gId*cl::sycl::id(1,NUM_VARS);
            // Physical position of cell center
            Real xc = dx*(Real(idx[0]) + Real(0.5)) + xLo;
            Real yc = dy*(Real(idx[1]) + Real(0.5)) + yLo;
#if (SPACE_DIM == 3)
            Real zc = dz*(Real(idx[2]) + Real(0.5)) + zLo;
#endif
            // Get reference to quadrature rule
            auto ab = QuadRule<NUM_GAUSS_POINTS>::ab;
            auto wt = QuadRule<NUM_GAUSS_POINTS>::wt;
            
            // Call IC function and accumulate integral
            Real pos[SPACE_DIM];
            Real work[NUM_VARS];
            Real avg[NUM_VARS];
            std::memset(avg,0,NUM_VARS*sizeof(Real));
#if (SPACE_DIM == 3)
            for (std::size_t nZ=0; nZ<NUM_GAUSS_POINTS; ++nZ) {
              pos[2] = sclAb(zc,dz,ab[nZ]);
#endif
              for (std::size_t nY=0; nY<NUM_GAUSS_POINTS; ++nY) {
                pos[1] = sclAb(yc,dy,ab[nY]);
                for (std::size_t nX=0; nX<NUM_GAUSS_POINTS; ++nX) {
                  pos[0] = sclAb(xc,dx,ab[nX]);
                  InitialCondition(fProp,pos,work);
                  for (std::size_t nV=0; nV<NUM_VARS; ++nV) {
                    // Accumulate into result
                    // Note no scaling by del here so that this forms a cell average
#if (SPACE_DIM == 3)
                    // 0.125 factor needed since (-1,1)x(-1,1)x(-1,1) has volume 8
                    avg[nV] += Real(0.125)*wt[nX]*wt[nY]*wt[nZ]*work[nV];
#else
                    // 0.25 factor needed since (-1,1)x(-1,1) has area 4
                    avg[nV] += Real(0.25)*wt[nX]*wt[nY]*work[nV];
#endif
                  }
                }
              }
#if (SPACE_DIM == 3)
            }
#endif
            // Write cell average out
            std::memcpy(&acc[dst],avg,NUM_VARS*sizeof(Real));
          });
      });
    }

  } // End numeric namespace
} // End KFVM namespace

#endif
