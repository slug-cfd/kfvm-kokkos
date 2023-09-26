// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// ICs from pluto AMR paper w/o B field
// see doi:10.1088/0067-0049/198/1/7

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::cosh;
  using Kokkos::exp;
  using Kokkos::pow;
  using Physics::Hydro::SimVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoH = userParams.rhoH, rhoL = userParams.rhoL;
  const Real rho = y <= 0.0 ? rhoL : rhoH;
  const Real p = 100.0 / eosParams.gamma - rho * y;
  const Real v = -0.1 * exp(-25.0 * (x * x + z * z)) / pow(cosh(100.0 * y * y), 2.0);

  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / gamm + 0.5 * rho * v * v;
}
