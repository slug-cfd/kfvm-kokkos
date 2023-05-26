#ifndef SIMVAR_H_
#define SIMVAR_H_

#include "../Types.H"

namespace KFVM {

  namespace Physics {

    namespace Hydro {

      struct SimVar {
        enum var { dens = 0, momx, momy, momz, etot};
      };
      struct PrimVar {
        enum var { dens = 0, velx, vely, velz, pres};
      };
      struct AuxVar {
        enum var { velx = 0, vely, velz, eint, pres};
      };

      typedef Real ConsVec[5];
      typedef Real PrimVec[5];

    } // end namespace Hydro

    namespace MHD_GLM {

      struct SimVar {
        enum var { dens = 0, momx, momy, momz, etot, magx, magy, magz, psi };
      };
      struct PrimVar {
        enum var { dens = 0, velx, vely, velz, prsg, magx, magy, magz, psi };
      };
      struct AuxVar {
        enum var { velx = 0, vely, velz, eint, pres, prsg, prsb };
      };

      typedef Real ConsVec[9];
      typedef Real PrimVec[9];

    } // end namespace MHD_GLM

    namespace SRHydro {

      struct SimVar {
        enum var { dens = 0, momx, momy, momz, tau};
      };
      struct PrimVar {
        enum var { rho = 0, velx, vely, velz, lorz, pres};
      };
      struct AuxVar {
        enum var { rho = 0, velx, vely, velz, lorz, pres};
      };

      typedef Real ConsVec[5];
      typedef Real PrimVec[6];

    } // end namespace SRHydro

  } // end namespace Physics

} // end namespace KFVM

#endif
