// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sin;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gam = eosParams.gamma;
  const Real gamm = gam - 1.0;
  const Real rho = gam * gam, p = gam, eps = 0.2;

  const Real u = -(1.0 + eps * sin(2.0 * M_PI * z)) * sin(2.0 * M_PI * y);
  const Real v = (1.0 + eps * sin(2.0 * M_PI * z)) * sin(2.0 * M_PI * x);
  const Real w = eps * sin(2.0 * M_PI * z);

  const Real bx = -sin(2.0 * M_PI * y), by = sin(4.0 * M_PI * x), bz = 0.0;

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
