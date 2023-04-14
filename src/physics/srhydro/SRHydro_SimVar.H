#ifndef SRHYDRO_SIMVAR_H_
#define SRHYDRO_SIMVAR_H_

#include "../../Types.H"

namespace KFVM {

  namespace Physics {

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
