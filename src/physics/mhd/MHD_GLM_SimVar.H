#ifndef MHD_GLM_SIMVAR_H_
#define MHD_GLM_SIMVAR_H_

namespace KFVM {

  namespace Physics {

    namespace MHD_GLM {
      
      enum SimVar { dens = 0, momx, momy, momz, etot, magx, magy, magz, psi };
      enum AuxVar { velx = 0, vely, velz, eint, pres, prsg, prsb };

    } // end namespace MHD_GLM

  } // end namespace Physics

} // end namespace KFVM

#endif
