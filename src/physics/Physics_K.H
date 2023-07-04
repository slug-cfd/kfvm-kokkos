#ifndef PHYSICS_K_H_
#define PHYSICS_K_H_

#include "EquationTypes.H"
#include "RSTypes.H"
#include "../Types.H"

namespace KFVM {

  namespace Physics {

    // Empty primary templates for each needed functor struct
    template<EquationType ET,class UViewType> struct AuxVars {};
    template<EquationType ET> class VarConvert {};
    template<EquationType ET,RSType RS> struct RiemannSolverX_K {};
    template<EquationType ET,RSType RS> struct RiemannSolverY_K {};
    template<EquationType ET,RSType RS> struct RiemannSolverZ_K {};
    template<EquationType ET> struct SpeedEstimate_K {};
    template<EquationType ET,class UViewType> struct SourceTerms_K {};
    template<EquationType ET,class UViewType> struct PositivityPreserve_K {};
    template<EquationType ET> struct EOS {};
    template<EquationType ET> struct BCCoeffs {};

  } // end namespace Physics
  
} // end namespace KFVM

// Include the specific physics kernels to provide specializations of the above

// Newtonian hydrodynamics (Adiabatic Euler)
#include "hydro/Hydro_BCCoeffs.H"
#include "hydro/Hydro_EOS.H"
#include "hydro/Hydro_K.H"

// Ideal MHD with GLM
#include "mhd/MHD_GLM_BCCoeffs.H"
#include "mhd/MHD_GLM_EOS.H"
#include "mhd/MHD_GLM_K.H"

// Special relativistic hydrodynamics (Adiabatic Euler)
#include "srhydro/SRHydro_BCCoeffs.H"
#include "srhydro/SRHydro_EOS.H"
#include "srhydro/SRHydro_K.H"

// Linear advection for simple tests
#include "linadv/LinAdv_BCCoeffs.H"
#include "linadv/LinAdv_EOS.H"
#include "linadv/LinAdv_K.H"

#endif
