// File: Parameterstructs.cpp
// Purpose: Provide definitions of functions that fill structs,
//          mostly as relevant for the forcing part of EOSparameters

#include <cmath>
#include <random>

#include <mpi.h>

#include "ParameterStructs.H"

namespace KFVM {

// Main setter function to interface with init file reader
bool EosParameters::set(const std::string &fld, const std::string &val) {
  std::size_t pos;

  if (fld == "gamma") {
    gamma = std::stof(val, &pos);
  } else if (fld == "reynolds") {
    Re = std::stof(val, &pos);
  } else if (fld == "prandtl") {
    Pr = std::stof(val, &pos);
  } else if (fld == "wenothresh") {
    wenoThresh = std::stof(val, &pos);
  } else if (fld == "forceedot") {
    forceEDot = std::stof(val, &pos);
  } else if (fld == "forcestr") {
    forceStr = std::stof(val, &pos);
  } else if (fld == "forcetoff") {
    forceTOff = std::stof(val, &pos);
  } else if (fld == "forceratio") {
    forceRatio = std::stof(val, &pos);
  } else {
    return false;
  }

  return true;
}

void EosParameters::updateForcing(const Real dt, int rank, int size,
                                  const MPI_Datatype dtype, MPI_Comm comm) {
#ifdef ENABLE_RANDOM_FORCING
  // Host mirror of forcing view
  auto hf = Kokkos::create_mirror(fAmp);

  if (rank == 0) { // only form on rank 0
    // random device and generator
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<Real> sn{0.0, 1.0};

    // copy down to modify host-side
    Kokkos::deep_copy(hf, fAmp);

    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      // extract wavevector and pad out 3rd component in 2d
      const Real k1 = hf(n, 0, 0), k2 = hf(n, 1, 0);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0;
#else
      const Real k3 = hf(n, 2, 0);
#endif

      // Get std dev for this k-vector
      const Real kMag = std::sqrt(k1 * k1 + k2 * k2 + k3 * k3);
      const Real sig = std::sqrt(kMag * (4.0 - kMag));

      for (int d = 0; d < SPACE_DIM; d++) {
        hf(n, d, 1) = sig * sn(gen); // real
        hf(n, d, 2) = sig * sn(gen); // imag
      }
    }
  }
  Kokkos::fence();

  // Broadcast host mirror to other ranks
  if (size > 1) {
    MPI_Bcast(hf.data(), (int)hf.size(), dtype, 0, comm);
  }

  // Copy back to device
  Kokkos::deep_copy(fAmp, hf);
#endif
}

} // end namespace KFVM
