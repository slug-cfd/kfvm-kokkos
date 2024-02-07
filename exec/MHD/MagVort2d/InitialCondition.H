// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// taken from doi:10.1016/j.jcp.2010.04.013 with q==1

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::exp;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real tpi = 2.0 * M_PI, kappa = 1.0 / tpi, mu = 1.0 / tpi, rsq = x * x + y * y;
  const Real ex1 = exp(1.0 - rsq), ex2 = exp(2.0 * (1.0 - rsq));

  const Real rho = 1.0;

  const Real u = 1.0 - y * kappa * ex1;
  const Real v = 1.0 + x * kappa * ex1;
  const Real w = 0.0;

  const Real bx = -y * mu * ex1;
  const Real by = x * mu * ex1;
  const Real bz = 0.0;

  const Real p = 1.0 + (mu * mu * (1.0 - 2 * rsq) - kappa * kappa) * ex2 / 4.0;

  state[ConsVar::dens] = rho;

  state[ConsVar::momx] = rho * u;
  state[ConsVar::momy] = rho * v;
  state[ConsVar::momz] = rho * w;

  state[ConsVar::magx] = bx;
  state[ConsVar::magy] = by;
  state[ConsVar::magz] = bz;

  state[ConsVar::etot] = p / gamm + rho * (u * u + v * v + w * w) / 2.0 +
                        (bx * bx + by * by + bz * bz) / 2.0;
}
