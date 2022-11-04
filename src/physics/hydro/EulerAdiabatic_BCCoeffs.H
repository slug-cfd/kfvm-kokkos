#ifndef EULERADIABATIC_BCCOEFFS_H_
#define EULERADIABATIC_BCCOEFFS_H_

#include <array>

#include <Kokkos_Core.hpp>

#include "../EquationTypes.H"
#include "../../Types.H"

namespace KFVM {

  namespace Physics {

    namespace EulerAdiabatic_BCCoeff_LUT {
      static constexpr Real refX[] = {1.0,-1.0,1.0,1.0,1.0};
      static constexpr Real refY[] = {1.0,1.0,-1.0,1.0,1.0};
      static constexpr Real refZ[] = {1.0,1.0,1.0,-1.0,1.0};
    }

    template<>
    struct BCCoeffs<EquationType::Euler>
    {
      Kokkos::View<Real[5],MemSpace> x,y,z;
      
      BCCoeffs<EquationType::Euler>():
        x("Physics::BCCoeffs::Reflect::x"),
        y("Physics::BCCoeffs::Reflect::y"),
        z("Physics::BCCoeffs::Reflect::z")
      {
        auto h_x = Kokkos::create_mirror(x);
        auto h_y = Kokkos::create_mirror(y);
        auto h_z = Kokkos::create_mirror(z);
        
        for (int n=0; n<5; n++) {
          h_x(n) = EulerAdiabatic_BCCoeff_LUT::refX[n];
          h_y(n) = EulerAdiabatic_BCCoeff_LUT::refY[n];
          h_z(n) = EulerAdiabatic_BCCoeff_LUT::refZ[n];
        }
        
        Kokkos::deep_copy(x,h_x);
        Kokkos::deep_copy(y,h_y);
        Kokkos::deep_copy(z,h_z);
      }
    };

  } // end namespace Physics
  
} // end namespace KFVM

#endif
