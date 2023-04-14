#ifndef HYDRO_SIMVAR_H_
#define HYDRO_SIMVAR_H_

#include "../../Types.H"

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

  } // end namespace Physics

} // end namespace KFVM

#endif
