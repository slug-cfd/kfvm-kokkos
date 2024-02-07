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

  const Real gamm = eosParams.gamma - 1.0;
  const Real Rsq = userParams.R * userParams.R;
  const Real rsq = SPACE_DIM == 2 ? x * x + y * y : x * x + y * y + z * z;
  const Real pG = rsq <= Rsq ? userParams.pIn : userParams.pOut;

  const Real rho = 1.0, bx = userParams.bx;

  state[ConsVar::dens] = rho;

  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;

  state[ConsVar::magx] = bx;
  state[ConsVar::magy] = 0.0;
  state[ConsVar::magz] = 0.0;

  state[ConsVar::etot] = pG / gamm + 0.5 * bx * bx;
}
