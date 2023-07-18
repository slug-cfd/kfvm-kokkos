// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::sqrt;
  using Physics::Hydro::SimVar;

  (void)x;
  (void)y;
  (void)z;

  const Real rhoH = 10.0, rhoL = 1.0;
  const Real R1 = 2.0, R2 = 4.0, P0 = 10.0, A = 0.1;

  const Real r = sqrt(x * x + y * y + z * z), phi = atan2(y, x);

  Real rho;

  // Density with perturbation
  const Real Ri = R1 + A * cos(12.0 * phi);
  if (r <= Ri) {
    rho = rhoL;
  } else if (r <= R2) {
    rho = rhoH;
  } else {
    rho = rhoL;
  }

  Real p = r < 1.2 ? P0 : 1.0;

  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / (eosParams.gamma - 1.0);
}
