// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Kokkos::atan2;
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;
  using Kokkos::sqrt;
  using ConsVar = typename SimVar<EquationType::Hydro>::ConsVar;

  const Real rxy = sqrt(x * x + y * y), rsq = x * x + y * y + z * z;
  const Real th = atan2(rxy, z), phi = atan2(y, x);
  const Real A = userParams.A, M = userParams.M;
  const Real Rsq1 = pow(2.0 / 3.0 + A * cos(M * phi) * sin(th), 2), Rsq2 = 0.64;
  const Real rhoIn = 0.05, rhoMid = 1.0, rhoOut = 0.1;
  const Real eiIn = 0.15, eiMid = 0.15, eiOut = 15.0;

  state[ConsVar::dens] = rsq <= Rsq1 ? rhoIn : (rsq <= Rsq2 ? rhoMid : rhoOut);
  state[ConsVar::etot] = rsq <= Rsq1 ? eiIn : (rsq <= Rsq2 ? eiMid : eiOut);

  state[ConsVar::momx] = 0.0;
  state[ConsVar::momy] = 0.0;
  state[ConsVar::momz] = 0.0;
}
