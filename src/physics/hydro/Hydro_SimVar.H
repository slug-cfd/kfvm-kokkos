#ifndef HYDRO_SIMVAR_H_
#define HYDRO_SIMVAR_H_

namespace KFVM {

  namespace Physics {

    namespace Hydro {
      
      enum SimVar { dens = 0, momx, momy, momz, etot};
      enum AuxVar { velx = 0, vely, velz, eint, pres};

    } // end namespace Hydro

  } // end namespace Physics

} // end namespace KFVM

#endif
