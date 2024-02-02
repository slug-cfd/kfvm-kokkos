// Set source terms
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void SourceTerm(Real t, Real x, Real y, Real z, EosParameters eosParams,
                UserParameters userParams, const Real state[NUM_VARS],
                Real force[NUM_VARS]) {
  using Hydro::SimVar;
  using Kokkos::cos;
  using Kokkos::fabs;
  using Kokkos::sin;

  (void)t;
  (void)x;
  (void)y;
  (void)z;
  (void)eosParams;
  (void)userParams;
  (void)state;
  (void)force;

  const Real str = 2.0 * eosParams.forceStr; // Extra 2 to enforce conj. symm.
  const Real rho = state[SimVar::dens];
  const Real u = state[SimVar::momx] / rho, v = state[SimVar::momy] / rho,
             w = state[SimVar::momz] / rho;

  Real fu = 0.0, fv = 0.0, fw = 0.0;
#ifdef ENABLE_RANDOM_FORCING // Check so that forcing can be turned off from ckpt
  if (t < eosParams.forceTOff) {
    for (int n = 0; n < NUM_FORCING_MODES; n++) {
#if (SPACE_DIM == 2)
      const Real kdx = eosParams.fAmp(n, 0, 0) * x + eosParams.fAmp(n, 1, 0) * y;
#else
      const Real kdx = eosParams.fAmp(n, 0, 0) * x + eosParams.fAmp(n, 1, 0) * y +
                       eosParams.fAmp(n, 2, 0) * z;
#endif
      fu += eosParams.fAmp(n, 0, 1) * cos(2.0 * M_PI * kdx) -
            eosParams.fAmp(n, 0, 2) * sin(2.0 * M_PI * kdx);
      fv += eosParams.fAmp(n, 1, 1) * cos(2.0 * M_PI * kdx) -
            eosParams.fAmp(n, 1, 2) * sin(2.0 * M_PI * kdx);
#if (SPACE_DIM == 3)
      fw += eosParams.fAmp(n, 2, 1) * cos(2.0 * M_PI * kdx) -
            eosParams.fAmp(n, 2, 2) * sin(2.0 * M_PI * kdx);
#endif
    }
  }
#endif

  force[SimVar::momx] = str * rho * fu;
  force[SimVar::momy] = str * rho * fv;
  force[SimVar::momz] = str * rho * fw;
  force[SimVar::etot] = str * rho * (u * fu + v * fv + w * fw);
}
