#ifndef SRHYDRO_SIMVAR_H_
#define SRHYDRO_SIMVAR_H_

namespace KFVM {

  namespace Physics {

    namespace SRHydro {
      
      enum SimVar { dens = 0, momx, momy, momz, etot};
      enum AuxVar { velx = 0, vely, velz, eint, pres};

    } // end namespace SRHydro

  } // end namespace Physics

} // end namespace KFVM

#endif
