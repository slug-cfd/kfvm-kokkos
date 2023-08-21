#ifndef LINADV_BCCOEFFS_H_
#define LINADV_BCCOEFFS_H_

#include <array>

#include <Kokkos_Core.hpp>

#include "../../Types.H"
#include "../EquationTypes.H"

namespace KFVM {

namespace Physics {

template <>
struct BCCoeffs<EquationType::LinAdv> {
  Kokkos::View<Real[5], MemSpace> reflectX, reflectY, reflectZ;
  Kokkos::View<Real[5], MemSpace> noslipX, noslipY, noslipZ;

  BCCoeffs<EquationType::LinAdv>()
      : reflectX("Physics::BCCoeffs::reflectX"), reflectY("Physics::BCCoeffs::reflectY"),
        reflectZ("Physics::BCCoeffs::reflectZ"), noslipX("Physics::BCCoeffs::noslipX"),
        noslipY("Physics::BCCoeffs::noslipY"), noslipZ("Physics::BCCoeffs::noslipZ") {
    auto h_rx = Kokkos::create_mirror(reflectX);
    auto h_ry = Kokkos::create_mirror(reflectY);
    auto h_rz = Kokkos::create_mirror(reflectZ);

    auto h_nx = Kokkos::create_mirror(noslipX);
    auto h_ny = Kokkos::create_mirror(noslipY);
    auto h_nz = Kokkos::create_mirror(noslipZ);

    h_rx(0) = 1.0;
    h_ry(0) = 1.0;
    h_rz(0) = 1.0;
    h_nx(0) = 1.0;
    h_ny(0) = 1.0;
    h_nz(0) = 1.0;

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
