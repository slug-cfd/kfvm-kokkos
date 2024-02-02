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
  } else if (fld == "forcestr") {
    forceStr = std::stof(val, &pos);
  } else if (fld == "forcetdyn") {
    forceTDyn = std::stof(val, &pos);
  } else if (fld == "forcetoff") {
    forceTOff = std::stof(val, &pos);
  } else if (fld == "forceratio") {
    forceRatio = std::stof(val, &pos);
  } else {
    return false;
  }

  return true;
}

// updates forcing using Ornstein-Uhlenbeck process
void EosParameters::updateForcing(const Real dt, int rank, int size,
                                  const MPI_Datatype dtype, MPI_Comm comm) {
#ifdef ENABLE_RANDOM_FORCING
  // Host mirror of forcing view
  auto hf = Kokkos::create_mirror(fAmp);

  if (rank == 0) { // only form on rank 0
    // random device and generator
    std::random_device rd{};
    std::mt19937 gen{rd()};

    // Distribution type
    std::normal_distribution<Real> sn{0.0, 1.0};

    // Forcing type
    const Real fr = forceRatio, mfr = (1.0 - 2.0 * forceRatio);

    // Scalings from OU process
    // use full variance for first draw (dt == 0) then OU after
    const Real mFac = std::exp(-dt / forceTDyn);
    const Real sFac = dt > 0.0 ? std::sqrt(1.0 - std::exp(-2.0 * dt / forceTDyn)) : 1.0;

    // copy down to modify host-side
    Kokkos::deep_copy(hf, fAmp);

    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      // extract wavevector and pad out 3rd component in 2d
      const Real k1 = hf(n, 0, 0), k2 = hf(n, 1, 0);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0, awr = 0.0, awi = 0.0;
#else
      const Real k3 = hf(n, 2, 0);
#endif
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, kMag = std::sqrt(ksq);
      const Real sig =
          sFac * std::sqrt(std::pow(4.0 * M_PI * M_PI * ksq, 3) * std::exp(-4.0 * kMag));

      // Propagate forward by OU
      const Real aur = mFac * hf(n, 0, 1) + sig * sn(gen);
      const Real aui = mFac * hf(n, 0, 2) + sig * sn(gen);
      const Real avr = mFac * hf(n, 1, 1) + sig * sn(gen);
      const Real avi = mFac * hf(n, 1, 2) + sig * sn(gen);
#if (SPACE_DIM == 3)
      const Real awr = mFac * hf(n, 2, 1) + sig * sn(gen);
      const Real awi = mFac * hf(n, 2, 2) + sig * sn(gen);
#endif

      // Project onto solenoidal vs compressive
      const Real iksq = 1.0 / ksq;
      hf(n, 0, 1) = (fr + iksq * mfr * k1 * k1) * aur +
                    iksq * mfr * (k1 * k2 * avr + k1 * k3 * awr);

      hf(n, 0, 2) = (fr + iksq * mfr * k1 * k1) * aui +
                    iksq * mfr * (k1 * k2 * avi + k1 * k3 * awi);

      hf(n, 1, 1) = (fr + iksq * mfr * k2 * k2) * avr +
                    iksq * mfr * (k1 * k2 * aur + k2 * k3 * awr);

      hf(n, 1, 2) = (fr + iksq * mfr * k2 * k2) * avi +
                    iksq * mfr * (k1 * k2 * aui + k2 * k3 * awi);
#if (SPACE_DIM == 3)
      hf(n, 2, 1) = (fr + iksq * mfr * k3 * k3) * awr +
                    iksq * mfr * (k1 * k3 * aur + k2 * k3 * avr);

      hf(n, 2, 2) = (fr + iksq * mfr * k3 * k3) * awi +
                    iksq * mfr * (k1 * k3 * aui + k2 * k3 * avi);
#endif
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
