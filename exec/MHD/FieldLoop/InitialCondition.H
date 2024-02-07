// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::MHD_GLM>::ConsVar;
  (void)z;

  const Real gamm = eosParams.gamma - 1.0;
  const Real rho = 1.0, p = 1.0, A0 = 0.001;
  const Real R = 0.3, r = sqrt(x * x + y * y);

  const Real u = 2.0;
  const Real v = 1.0;
  const Real w = SPACE_DIM == 2 ? 0.0 : 0.1;

  const Real bx = (r < R ? -y * A0 / r : 0.0), by = (r < R ? x * A0 / r : 0.0), bz = 0.0;

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
