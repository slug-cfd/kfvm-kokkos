#ifndef MHD_GLM_SIMVAR_H_
#define MHD_GLM_SIMVAR_H_

#include "../../Types.H"

namespace KFVM {

  namespace Physics {

    namespace MHD_GLM {

      struct SimVar {
        enum var { dens = 0, momx, momy, momz, etot, magx, magy, magz, psi };
      };
      struct PrimVar {
        enum var { dens = 0, velx, vely, velz, pres, magx, magy, magz, psi };
      };
      struct AuxVar {
        enum var { velx = 0, vely, velz, eint, pres, prsg, prsb };
      };

      typedef Real ConsVec[9];
      typedef Real PrimVec[9];

    } // end namespace MHD_GLM

  } // end namespace Physics

} // end namespace KFVM

#endif
