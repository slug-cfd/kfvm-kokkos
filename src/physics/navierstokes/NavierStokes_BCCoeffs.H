#ifndef NAVIERSTOKES_BCCOEFFS_H_
#define NAVIERSTOKES_BCCOEFFS_H_

#include <array>

#include <Kokkos_Core.hpp>

#include "../../Types.H"
#include "../EquationTypes.H"

namespace KFVM {

namespace Physics {

namespace NavierStokes_BCCoeff_LUT {
static constexpr Real refX[] = {1.0, -1.0, 1.0, 1.0, 1.0};
static constexpr Real refY[] = {1.0, 1.0, -1.0, 1.0, 1.0};
static constexpr Real refZ[] = {1.0, 1.0, 1.0, -1.0, 1.0};

static constexpr Real noslip[] = {1.0, -1.0, -1.0, -1.0, 1.0};
} // namespace NavierStokes_BCCoeff_LUT

template <>
struct BCCoeffs<EquationType::NavierStokes> {
  Kokkos::View<Real[5], MemSpace> reflectX, reflectY, reflectZ;
  Kokkos::View<Real[5], MemSpace> noslipX, noslipY, noslipZ;

  BCCoeffs<EquationType::NavierStokes>()
      : reflectX("Physics::BCCoeffs::reflectX"), reflectY("Physics::BCCoeffs::reflectY"),
        reflectZ("Physics::BCCoeffs::reflectZ"), noslipX("Physics::BCCoeffs::noslipX"),
        noslipY("Physics::BCCoeffs::noslipY"), noslipZ("Physics::BCCoeffs::noslipZ") {
    auto h_rx = Kokkos::create_mirror(reflectX);
    auto h_ry = Kokkos::create_mirror(reflectY);
    auto h_rz = Kokkos::create_mirror(reflectZ);

    auto h_nx = Kokkos::create_mirror(noslipX);
    auto h_ny = Kokkos::create_mirror(noslipY);
    auto h_nz = Kokkos::create_mirror(noslipZ);

    for (int n = 0; n < 5; n++) {
      h_rx(n) = NavierStokes_BCCoeff_LUT::refX[n];
      h_ry(n) = NavierStokes_BCCoeff_LUT::refY[n];
      h_rz(n) = NavierStokes_BCCoeff_LUT::refZ[n];

      h_nx(n) = NavierStokes_BCCoeff_LUT::noslip[n];
      h_ny(n) = NavierStokes_BCCoeff_LUT::noslip[n];
      h_nz(n) = NavierStokes_BCCoeff_LUT::noslip[n];
    }

    Kokkos::deep_copy(reflectX, h_rx);
    Kokkos::deep_copy(reflectY, h_ry);
    Kokkos::deep_copy(reflectZ, h_rz);

    Kokkos::deep_copy(noslipX, h_nx);
    Kokkos::deep_copy(noslipY, h_ny);
    Kokkos::deep_copy(noslipZ, h_nz);
  }
};

} // end namespace Physics

} // end namespace KFVM

#endif
