#ifndef PHYSICS_K_H_
#define PHYSICS_K_H_

#include "EquationTypes.H"
#include "../Types.H"

namespace KFVM {

  namespace Physics {

    // Empty primary templates for each needed functor struct
    template<EquationType ET,class UViewType> struct AuxVars {};
    template<EquationType ET,class StateType> struct VarConvert {};
    template<EquationType ET> struct RiemannSolverX_K {};
    template<EquationType ET> struct RiemannSolverY_K {};
    template<EquationType ET> struct RiemannSolverZ_K {};
    template<EquationType ET> struct SpeedEstimate_K {};
    template<EquationType ET,class UViewType,class RHSViewType> struct SourceTerms_K {};
    template<EquationType ET,class UViewType> struct PositivityPreserve_K {};
    template<EquationType ET> struct EOS {};
    template<EquationType ET> struct BCCoeffs {};

  } // end namespace Physics
  
} // end namespace KFVM

// Include the specific physics kernels to provide specializations of the above

// Adiabatic Euler hydrodynamics
#include "hydro/EulerAdiabatic_BCCoeffs.H"
#include "hydro/EulerAdiabatic_EOS.H"
#include "hydro/EulerAdiabatic_K.H"

// Ideal MHD with GLM
#include "mhd/MHD_GLM_BCCoeffs.H"
#include "mhd/MHD_GLM_EOS.H"
#include "mhd/MHD_GLM_K.H"

#endif
