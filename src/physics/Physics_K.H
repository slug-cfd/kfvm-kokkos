#ifndef PHYSICS_K_H_
#define PHYSICS_K_H_

#include "EquationTypes.H"

namespace KFVM {

  namespace Physics {

    // Empty primary templates for each needed functor struct
    template<EquationType ET,class UViewType> struct AuxVars {};
    template<EquationType ET,class StateType> struct VarConvert {};
    template<EquationType ET> struct RiemannSolverX_K {};
    template<EquationType ET> struct RiemannSolverY_K {};
    template<EquationType ET> struct RiemannSolverZ_K {};
    template<EquationType ET,class UViewType> struct PositivityPreserve_K {};

  } // end namespace Physics
  
} // end namespace KFVM

// Include the specific physics kernels to provide specializations of the above
#include "hydro/Hydro_K.H"

#endif
