// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, EosParameters eosParams,
                UserParameters userParams, const Real state[NUM_VARS],
                Real force[NUM_VARS]) {
  using Kokkos::cos;
  using Kokkos::sin;
  using MHD_GLM::SimVar;

  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;
  (void)state;
  (void)force;

  const Real str = 2.0 * eosParams.forceStr; // Extra 2 to enforce conj. symm.
  const Real fr = eosParams.forceRatio, mfr = 1.0 - 2.0 * fr;
  const Real rho = state[SimVar::dens];
  const Real u = state[SimVar::momx] / rho, v = state[SimVar::momy] / rho,
             w = state[SimVar::momz] / rho;

  Real fu = 0.0, fv = 0.0, fw = 0.0;
#ifdef ENABLE_RANDOM_FORCING // Check so that forcing can be turned off from ckpt
  if (t < eosParams.forceTOff) {
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
      // pull out k-vector
      const Real k1 = eosParams.fAmp(n, 0, 0), k2 = eosParams.fAmp(n, 1, 0);
#if (SPACE_DIM == 2)
      const Real k3 = 0.0;
#else
      const Real k3 = eosParams.fAmp(n, 2, 0);
#endif

      // Set k.x and set forcing from this mode ignoring projection
      const Real kdx = k1 * x + k2 * y + k3 * z;
      const Real fun = eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      const Real fvn = eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 2)
      const Real fwn = 0.0;
#else
      const Real fwn = eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
                       eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif

      // Project and add into total forcing
      const Real ksq = k1 * k1 + k2 * k2 + k3 * k3, iksq = 1.0 / ksq;

      fu += (fr + mfr * iksq * k1 * k1) * fun +
            mfr * iksq * (k1 * k2 * fvn + k1 * k3 * fwn);

      fv += (fr + mfr * iksq * k2 * k2) * fvn +
            mfr * iksq * (k1 * k2 * fun + k2 * k3 * fwn);

      fw += (fr + mfr * iksq * k3 * k3) * fwn +
            mfr * iksq * (k1 * k3 * fun + k2 * k3 * fvn);
    }
  }
#endif

  force[SimVar::momx] = str * rho * fu;
  force[SimVar::momy] = str * rho * fv;
  force[SimVar::momz] = str * rho * fw;
  force[SimVar::etot] = str * rho * (u * fu + v * fv + w * fw);
}
