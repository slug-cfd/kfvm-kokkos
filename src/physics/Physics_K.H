#ifndef PHYSICS_K_H_
#define PHYSICS_K_H_

#include "../Types.H"
#include "EquationTypes.H"
#include "RSTypes.H"

namespace KFVM {

namespace Physics {

// Empty primary templates for each needed functor struct
template <EquationType ET, class UViewType>
struct AuxVars {};
template <EquationType ET, class UViewType>
struct FlowStats {};
template <EquationType ET>
class VarConvert {};
template <EquationType ET>
struct SpeedEstimate_K {};
template <EquationType ET, class UViewType, class RViewType>
struct Viscosity_K {};
template <EquationType ET, class UViewType>
struct SourceTerms_K {};
template <EquationType ET>
struct EOS {};
template <EquationType ET>
struct BCCoeffs {};

template <EquationType ET>
struct RSPhys {};

// Forward declare ones that have both generic and specialized versions
template <EquationType ET, RSType RST>
struct RiemannSolver;
template <RSType RST>
struct RiemannSolverX_K;
template <RSType RST>
struct RiemannSolverY_K;
template <RSType RST>
struct RiemannSolverZ_K;

template <EquationType ET, class UViewType>
struct PositivityPreserve_K;

template <EquationType ET, class UViewType>
struct RandForcingIntegrate_K;
template <EquationType ET, class UViewType>
struct RandForcingApply_K;

} // end namespace Physics

} // end namespace KFVM

// Include the specific physics kernels to provide specializations of the above

// Newtonian hydrodynamics (Adiabatic Euler)
#include "hydro/Hydro_BCCoeffs.H"
#include "hydro/Hydro_EOS.H"
#include "hydro/Hydro_K.H"
#include "hydro/Hydro_Riemann.H"

// Navier-Stokes (hydro + viscosity)
#include "navierstokes/NavierStokes_BCCoeffs.H"
#include "navierstokes/NavierStokes_EOS.H"
#include "navierstokes/NavierStokes_K.H"
#include "navierstokes/NavierStokes_Riemann.H"

// Ideal MHD with GLM
#include "mhd_glm/MHD_GLM_BCCoeffs.H"
#include "mhd_glm/MHD_GLM_EOS.H"
#include "mhd_glm/MHD_GLM_K.H"
#include "mhd_glm/MHD_GLM_Riemann.H"

// Ideal MHD with 8-wave
#include "mhd_8w/MHD_8W_BCCoeffs.H"
#include "mhd_8w/MHD_8W_EOS.H"
#include "mhd_8w/MHD_8W_K.H"
#include "mhd_8w/MHD_8W_Riemann.H"

// Special relativistic hydrodynamics (Adiabatic Euler)
#include "srhydro/SRHydro_BCCoeffs.H"
#include "srhydro/SRHydro_EOS.H"
#include "srhydro/SRHydro_K.H"
#include "srhydro/SRHydro_Riemann.H"

// Linear advection for simple tests
#include "linadv/LinAdv_BCCoeffs.H"
#include "linadv/LinAdv_EOS.H"
#include "linadv/LinAdv_K.H"
#include "linadv/LinAdv_Riemann.H"

// Common kernels for multiple equation types
#include "Common_K.H"

#endif
