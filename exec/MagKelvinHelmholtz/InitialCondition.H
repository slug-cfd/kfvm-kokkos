// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::exp;
  using Kokkos::sin;
  using Kokkos::tanh;
  using Physics::MHD_GLM::SimVar;

  (void)z;
  const Real gamm = eosParams.gamma - 1.0;

  Real sig = 0.2, A = 0.01, a = 0.05, y1 = 0.5, y2 = 1.5;
  Real rho = 1.0 + (tanh((y - y1) / a) - tanh((y - y2) / a)) / 2.0;
  Real u = (tanh((y - y1) / a) - tanh((y - y2) / a) - 1.0);
  Real v =
      A * sin(2.0 * M_PI * x) *
      (exp(-(y - y1) * (y - y1) / (sig * sig)) + exp(-(y - y2) * (y - y2) / (sig * sig)));
  Real b = 0.5 * (tanh((y - y1) / a) - tanh((y - y2) / a) - 1.0);

  state[SimVar::dens] = rho;

  state[SimVar::momx] = rho * u;
  state[SimVar::momy] = rho * v;
  state[SimVar::momz] = 0.0;

  state[SimVar::magx] = b;
  state[SimVar::magy] = 0.0;
  state[SimVar::magz] = 0.0;

  state[SimVar::etot] = 10.0 / gamm + b * b / 2.0 + rho * (u * u + v * v) / 2.0;

  state[SimVar::psi] = 0.0;
}
