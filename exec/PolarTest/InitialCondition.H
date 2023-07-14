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

  const Real r = sqrt(x * x + y * y), phi = atan2(y, x);
  const Real g = -1.0 / (r * r);
  const Real gx = g * cos(phi), gy = g * sin(phi);

  Real rho, p;

  // Density with perturbation
  const Real Ri = R1 + A * cos(12.0 * phi);
  if (r <= Ri) {
    rho = rhoL;
  } else if (r <= R2) {
    rho = rhoH;
  } else {
    rho = rhoL;
  }

  // Pressure hydrostatic ignoring density perturbation
  if (r <= 1.0) {
    p = P0 - 0.5 * rhoL * r;
  } else if (r <= R1) {
    p = P0 + rhoL * (1.0 / r - 1.5);
  } else if (r <= R2) {
    p = P0 + rhoL * (1.0 / R1 - 1.5) + rhoH * (1.0 / r - 1.0 / R1);
  } else {
    p = P0 + rhoL * (1.0 / r + 1.0 / R1 - 1.5 - 1.0 / R2) + rhoH * (1.0 / R2 - 1.0 / R1);
  }

  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / (eosParams.gamma - 1.0);
}
