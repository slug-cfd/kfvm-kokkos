// Set intial condition
// must take (x,y,z) coordinates regardless of space dimension

KOKKOS_INLINE_FUNCTION
void InitialCondition(Real x, Real y, Real z, EosParameters eosParams,
                      UserParameters userParams, Real state[NUM_VARS]) {
  using Physics::MHD_GLM::SimVar;

  // Parallel coordinate
  const Real xp = (2.0 * x + y) / sqrt(5.0);

  // "Left" state regions
  const bool left = (xp < 0.5) || (1.5 < xp && xp < 2.5) || (3.5 < xp);

  const Real gamm = eosParams.gamma - 1.0;
  const Real rhoL = 1.0, rhoR = 0.125, rho = left ? rhoL : rhoR;
  const Real pL = 1.0, pR = 0.1, p = left ? pL : pR;
  const Real bPar = 0.75, bPerp = left ? 1.0 : -1.0;
  const Real bx = (2.0 * bPar - bPerp) / sqrt(5.0);
  const Real by = (bPar - 2.0 * bPerp) / sqrt(5.0);

  state[SimVar::dens] = rho;
  state[SimVar::momx] = 0.0;
  state[SimVar::momy] = 0.0;
  state[SimVar::momz] = 0.0;
  state[SimVar::etot] = p / gamm + 0.5 * (bx * bx + by * by);
  state[SimVar::magx] = bx;
  state[SimVar::magy] = by;
  state[SimVar::magz] = 0.0;
}
