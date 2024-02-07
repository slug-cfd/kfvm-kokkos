// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::exp;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rsq = x * x + y * y + z * z, A0 = 20.0;
  const Real exr = exp(-rsq);

  const Real rho = 1.0, p = 1.0;

  const Real u = 0.0;
  const Real v = 0.0;
  const Real w = 0.0;

  const Real bx = (2.0 * x * z - A0 * y) * exr;
  const Real by = (2.0 * y * z + A0 * x) * exr;
  const Real bz = 2.0 * (1.0 - x * x - y * y) * exr;

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
