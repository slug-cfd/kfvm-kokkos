// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
Real ivOmega(Real x, Real y, Real xc, Real yc, Real R, Real sig, Real beta) {
  return beta *
         exp(-((x - xc) * (x - xc) + (y - yc) * (y - yc)) / (2.0 * sig * sig * R * R));
}

KOKKOS_INLINE_FUNCTION
void ivAddPert(Real x, Real y, Real xc, Real yc, Real R, Real Om, Real gamma,
               Real Ub[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  Real rhoB = Ub[SimVar::dens];
  Real presB = (gamma - 1.0) * (Ub[SimVar::etot] - (Ub[SimVar::momx] * Ub[SimVar::momx] +
                                                    Ub[SimVar::momy] * Ub[SimVar::momy]) /
                                                       (2.0 * rhoB));
  Ub[SimVar::dens] =
      pow(Ub[SimVar::dens] + (1.0 - gamma) * Om * Om / 2.0, 1.0 / (gamma - 1.0));
  Ub[SimVar::momx] = Ub[SimVar::dens] * (Ub[SimVar::momx] / rhoB - (y - yc) * Om / R);
  Ub[SimVar::momy] = Ub[SimVar::dens] * (Ub[SimVar::momy] / rhoB + (x - xc) * Om / R);
  Ub[SimVar::etot] =
      pow(presB + (1.0 - gamma) * Om * Om / 2.0, gamma / (gamma - 1.0)) /
          (gamma * (gamma - 1.0)) +
      (Ub[SimVar::momx] * Ub[SimVar::momx] + Ub[SimVar::momy] * Ub[SimVar::momy]) /
          (2.0 * Ub[SimVar::dens]);
}

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::Hydro::SimVar;

  (void)z;

  const Real gamm = eosParams.gamma - 1.0;

  Real xc = 0.0, yc = 0.0;
  Real R = 1.0, sig = 1.0, beta = 5.0 * sqrt(2.0) * exp(0.5) / (4.0 * M_PI);
  Real Om = ivOmega(x, y, xc, yc, R, sig, beta);

  state[SimVar::dens] = 1.0;
  state[SimVar::momx] = 1.0;
  state[SimVar::momy] = 1.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = 1.0 / gamm + 1.0;

  ivAddPert(x, y, xc, yc, R, Om, eosParams.gamma, state);
}
