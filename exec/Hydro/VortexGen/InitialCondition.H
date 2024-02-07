// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

// See doi:10.1006/jcph.2000.6606

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real gamm = eosParams.gamma - 1.0;

  const Real r1sq = x * x + y * y, r2sq = (x - 0.4) * (x - 0.4) + z * z;
  const Real Rsq = 0.2 * 0.2;

  const Real rho = r2sq < Rsq ? 0.1 : 1.0;
  const Real p = r1sq < Rsq ? 10.0 : 1.0;

  state[ConsVar::dens] = rho;
  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
  state[ConsVar::etot] = p / gamm;
}
