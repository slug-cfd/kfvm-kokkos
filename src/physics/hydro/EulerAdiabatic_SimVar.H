#ifndef EULERADIABATIC_SIMVAR_H_
#define EULERADIABATIC_SIMVAR_H_

namespace KFVM {

  namespace Physics {

    namespace EulerAdiabatic {
      
      enum SimVar { dens = 0, momx, momy, momz, etot};
      enum AuxVar { velx = 0, vely, velz, eint, pres};

    } // end namespace EulerAdiabatic

  } // end namespace Physics

} // end namespace KFVM

#endif
