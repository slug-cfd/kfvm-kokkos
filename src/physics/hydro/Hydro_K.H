#ifndef HYDRO_K_H_
#define HYDRO_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits>

#include "Definitions.H"
#include "../../Dimension.H"
#include "../../Types.H"
#include "../../Geometry.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "Hydro_SimVar.H"
#include "Hydro_EOS.H"
#include "Hydro_Riemann.H"

namespace KFVM {

  namespace Physics {

    template<class StateType>
    struct VarConvert<EquationType::Hydro,StateType>
    {
      KOKKOS_INLINE_FUNCTION
      static void Cons2Prim(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using Hydro::SimVar;
	
        Real gm = gam - Real(1.0);
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/Real(2.0);
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = gm*(vsq*state(SimVar::dens) -
                                  u*state(SimVar::momx) - 
                                  v*state(SimVar::momy) - 
                                  w*state(SimVar::momz) +
                                  state(SimVar::etot));
      
        state(SimVar::momx) = (state(SimVar::momx) - u*state(SimVar::dens))/rho;
        state(SimVar::momy) = (state(SimVar::momy) - v*state(SimVar::dens))/rho;
        state(SimVar::momz) = (state(SimVar::momz) - w*state(SimVar::dens))/rho;
      }
      
      KOKKOS_INLINE_FUNCTION
      static void Prim2Cons(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using Hydro::SimVar;
	
        Real gm = gam - Real(1.0);
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/Real(2.0);
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = (vsq*state(SimVar::dens) +
                               rho*u*state(SimVar::momx) + 
                               rho*v*state(SimVar::momy) + 
                               rho*w*state(SimVar::momz) +
                               state(SimVar::etot)/gm);
      
        state(SimVar::momx) = (rho*state(SimVar::momx) + u*state(SimVar::dens));
        state(SimVar::momy) = (rho*state(SimVar::momy) + v*state(SimVar::dens));
        state(SimVar::momz) = (rho*state(SimVar::momz) + w*state(SimVar::dens));
      }
    };

    template<class UViewType>
    struct AuxVars<EquationType::Hydro,UViewType>
    {
      UViewType U;
      AuxDataView V;
      FluidProperties fProp;

      AuxVars(const UViewType& U_,const AuxDataView V_,const FluidProperties& fProp_):
	U(U_),
        V(V_),
        fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
        using Hydro::SimVar;
        using Hydro::AuxVar;

	// Try not to assume the order of SimVars
	Real ua[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  ua[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	
	// Set velocities
	V(KFVM_D_DECL(i,j,k),AuxVar::velx) =
	  U(KFVM_D_DECL(i,j,k),SimVar::momx)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::vely) =
	  U(KFVM_D_DECL(i,j,k),SimVar::momy)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::velz) =
	  U(KFVM_D_DECL(i,j,k),SimVar::momz)/U(KFVM_D_DECL(i,j,k),SimVar::dens);

	// Internal energy and pressure
	V(KFVM_D_DECL(i,j,k),AuxVar::eint) = EOS<EquationType::Hydro>::eint(ua);
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = EOS<EquationType::Hydro>::pres(fProp.gamma,ua);
      }
    };

    template<RSType rst>
    struct RiemannSolverX_K<EquationType::Hydro,rst>
    {
      FaceDataView U;
      FluidProperties fProp;

      RiemannSolverX_K(const FaceDataView& U_,const FluidProperties& fProp_):
	U(U_),
        fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k),Real& maxVel) const
      {
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS],F[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }

	  // Calculate flux
	  Real vq = HydroRS<rst>::RiemannSolve_X(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(KFVM_D_DECL(i,j,k),0,nQ,nV) = F[nV];
	    U(KFVM_D_DECL(i,j,k),1,nQ,nV) = F[nV];
	  }
	}
      }
    };

    template<RSType rst>
    struct RiemannSolverY_K<EquationType::Hydro,rst>
    {
      FaceDataView U;
      FluidProperties fProp;

      RiemannSolverY_K(const FaceDataView& U_,const FluidProperties& fProp_):
	U(U_),
        fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k),Real& maxVel) const
      {
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS],F[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }

	  // Calculate flux
	  Real vq = HydroRS<rst>::RiemannSolve_Y(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	  
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(KFVM_D_DECL(i,j,k),0,nQ,nV) = F[nV];
	    U(KFVM_D_DECL(i,j,k),1,nQ,nV) = F[nV];
	  }
	}
      }
    };

    template<RSType rst>
    struct RiemannSolverZ_K<EquationType::Hydro,rst>
    {
      FaceDataView U;
      FluidProperties fProp;

      RiemannSolverZ_K(const FaceDataView& U_,const FluidProperties& fProp_):
	U(U_),
        fProp(fProp_)
      {}

#if (SPACE_DIM == 3)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS],F[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,k,0,nQ,nV);
	    uR[nV] = U(i,j,k,1,nQ,nV);
	  }
	  
	  // Calculate flux
	  Real vq = HydroRS<rst>::RiemannSolve_Z(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	  
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(i,j,k,0,nQ,nV) = F[nV];
	    U(i,j,k,1,nQ,nV) = F[nV];
	  }
	}
      }
#endif
    };

    template<>
    struct SpeedEstimate_K<EquationType::Hydro>
    {
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      FluidProperties fProp;

      SpeedEstimate_K(KFVM_D_DECL(const FaceDataView& rsX_,
				  const FaceDataView& rsY_,
				  const FaceDataView& rsZ_),const FluidProperties& fProp_):
	KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
        fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k),Real& maxVel) const
      {}
    };

    namespace Hydro {
#include "SourceTerms.H"
    }

    template<class UDataView,class RHSDataView>
    struct SourceTerms_K<EquationType::Hydro,UDataView,RHSDataView>
    {
      UDataView U;
      RHSDataView RHS;
      FluidProperties fProp;
      Geometry geom;
      Real t;

      SourceTerms_K(const UDataView& U_,const RHSDataView RHS_,
		    const FluidProperties& fProp_,const Geometry& geom_,const Real t_):
	U(U_),
	RHS(RHS_),
	fProp(fProp_),
	geom(geom_),
        t(t_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
        Real state[NUM_VARS],force[NUM_VARS];

        // extract state, zero out force
        for (int n=0; n<NUM_VARS; n++) {
          state[n] = U(KFVM_D_DECL(i,j,k),n);
          force[n] = 0.0;
        }

        // Call user-defined source terms
        Real xc = geom.xCoord(i);
        Real yc = geom.yCoord(j);
#if (SPACE_DIM == 2)
        Real zc = geom.zCoord(0);
#else
        Real zc = geom.zCoord(k);
#endif
        Hydro::SourceTerm(t,xc,yc,zc,state,force);

        // Apply to RHS
        for (int n=0; n<NUM_VARS; n++) {
          RHS(KFVM_D_DECL(i,j,k),n) += force[n];
        }
      }
    };

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::Hydro,UDataView>
    {
      UDataView U;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      FluidProperties fProp;

      PositivityPreserve_K(const UDataView& U_,
			   KFVM_D_DECL(const FaceDataView& rsX_,
				       const FaceDataView& rsY_,
				       const FaceDataView& rsZ_),
			   const FluidProperties& fProp_):
	U(U_),
	KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
	fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
	using Kokkos::fmin;
	using Kokkos::fmax;
	using Hydro::SimVar;

	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
	Real refState[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::Hydro>::pres(fProp.gamma,refState);
	
	// Find ranges of density, pressure, and soundspeed
	Real maxDens = -std::numeric_limits<Real>::max();
	Real minDens = std::numeric_limits<Real>::max();
	Real minPres = std::numeric_limits<Real>::max();
	Real minSdSp = std::numeric_limits<Real>::max();
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
#if (SPACE_DIM == 3)
	    for (idx_t zo=-1; zo<=1; zo++) {
#endif
	      Real state[NUM_VARS];
	      for (int nV=0; nV<NUM_VARS; nV++) {
		state[nV] = U(KFVM_D_DECL(i + xo,j + yo,k + zo),nV);
	      }
	      maxDens = fmax(maxDens,state[SimVar::dens]);
	      minDens = fmin(minDens,state[SimVar::dens]);
              Real pres = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
              Real sdsp = EOS<EquationType::Hydro>::soundspeed(fProp.gamma,state);
	      minPres = fmin(minPres,pres);
	      minSdSp = fmin(minSdSp,sdsp);
#if (SPACE_DIM == 3)
	    }
#endif
	  }
	}

        // Find size of flattener
        Real dxU = U(KFVM_D_DECL(i + 1,j,k),SimVar::momx)/U(KFVM_D_DECL(i + 1,j,k),SimVar::dens) -
	  U(KFVM_D_DECL(i - 1,j,k),SimVar::momx)/U(KFVM_D_DECL(i - 1,j,k),SimVar::dens);
        Real dyV = U(KFVM_D_DECL(i,j + 1,k),SimVar::momy)/U(KFVM_D_DECL(i,j + 1,k),SimVar::dens) -
	  U(KFVM_D_DECL(i,j - 1,k),SimVar::momy)/U(KFVM_D_DECL(i,j - 1,k),SimVar::dens);
#if (SPACE_DIM == 3)
        Real dzW = U(KFVM_D_DECL(i,j,k + 1),SimVar::momz)/U(KFVM_D_DECL(i,j,k + 1),SimVar::dens) -
	  U(KFVM_D_DECL(i,j,k - 1),SimVar::momz)/U(KFVM_D_DECL(i,j,k - 1),SimVar::dens);
#else
        Real dzW = Real(0.0);
#endif
        Real divV = (dxU + dyV + dzW)/Real(2.0); // Undivided difference
        Real kappa1 = Real(0.3),kappa2 = Real(0.3);
        Real eta = fmin(Real(1.0),fmax(Real(0.0),-(kappa1*minSdSp + divV)/(kappa1*minSdSp)));

        // Extend density and pressure ranges
        maxDens *= (Real(1.0) + kappa2 - kappa2*eta);
        minDens *= (Real(1.0) - kappa2 + kappa2*eta);
        minPres *= (Real(1.0) - kappa2 + kappa2*eta);
        
	Real rDensDiffP = maxDens - rDens,rDensDiffM = rDens - minDens;

	// Enforce positive density
	// Find mixing amount
	Real theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // West face
          Real rho = rsX(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
	  // East face
          rho = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
	  // South face
          rho = rsY(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
	  // North face
          rho = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
#if (SPACE_DIM == 3)
	  // Bottom face
          rho = rsZ(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
	  // Top face
          rho = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::dens);
	  theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
	  theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
#endif
	}
	
	// Apply mixing
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    rsX(KFVM_D_DECL(i    ,j,k),1,nQ,nV) = refState[nV] + theta*(rsX(KFVM_D_DECL(i    ,j,k),1,nQ,nV) - refState[nV]);
	    rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV) = refState[nV] + theta*(rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV) - refState[nV]);
	    rsY(KFVM_D_DECL(i,j    ,k),1,nQ,nV) = refState[nV] + theta*(rsY(KFVM_D_DECL(i,j    ,k),1,nQ,nV) - refState[nV]);
	    rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV) = refState[nV] + theta*(rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV) - refState[nV]);
#if (SPACE_DIM == 3)
	    rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,nV) = refState[nV] + theta*(rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,nV) - refState[nV]);
	    rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV) = refState[nV] + theta*(rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV) - refState[nV]);
#endif
	  }	  
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  Real p,state[NUM_VARS];
	  // West face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
	  // East face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
	  // South face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
	  // North face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
#if (SPACE_DIM == 3)
	  // Bottom face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
	  // Top face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV);
	  }
	  p = EOS<EquationType::Hydro>::pres(fProp.gamma,state);
          theta = p >= minPres ? theta : fmin(theta,eulerPresPos(fProp.gamma,minPres,refState,state));
#endif
	}
	
	// Apply mixing
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    rsX(KFVM_D_DECL(i    ,j,k),1,nQ,nV) = refState[nV] + theta*(rsX(KFVM_D_DECL(i    ,j,k),1,nQ,nV) - refState[nV]);
	    rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV) = refState[nV] + theta*(rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV) - refState[nV]);
	    rsY(KFVM_D_DECL(i,j    ,k),1,nQ,nV) = refState[nV] + theta*(rsY(KFVM_D_DECL(i,j    ,k),1,nQ,nV) - refState[nV]);
	    rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV) = refState[nV] + theta*(rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV) - refState[nV]);
#if (SPACE_DIM == 3)
	    rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,nV) = refState[nV] + theta*(rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,nV) - refState[nV]);
	    rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV) = refState[nV] + theta*(rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV) - refState[nV]);
#endif
	  }	  
	}
      }

      KOKKOS_INLINE_FUNCTION
      Real eulerPresPos(const Real gamma,const Real pmin,
			const Real refState[NUM_VARS],const Real state[NUM_VARS]) const
      {
	using Kokkos::sqrt;
	using Hydro::SimVar;

	Real emin = pmin/(gamma - Real(1.0));
      
	Real A = Real(2.0)*(state[SimVar::dens] - refState[SimVar::dens])*(state[SimVar::etot] - refState[SimVar::etot]);
	A -= ((state[SimVar::momx] - refState[SimVar::momx])*(state[SimVar::momx] - refState[SimVar::momx]) +
	      (state[SimVar::momy] - refState[SimVar::momy])*(state[SimVar::momy] - refState[SimVar::momy]) +
	      (state[SimVar::momz] - refState[SimVar::momz])*(state[SimVar::momz] - refState[SimVar::momz]));
      
	Real B = Real(2.0)*(refState[SimVar::dens]*(state[SimVar::etot] - refState[SimVar::etot]) +
			refState[SimVar::etot]*(state[SimVar::dens] - refState[SimVar::dens]));
	B -= Real(2.0)*((state[SimVar::momx] - refState[SimVar::momx])*refState[SimVar::momx] +
		    (state[SimVar::momy] - refState[SimVar::momy])*refState[SimVar::momy] +
		    (state[SimVar::momz] - refState[SimVar::momz])*refState[SimVar::momz]);
	B -= Real(2.0)*emin*(state[SimVar::dens] - refState[SimVar::dens]);
      
	Real C = Real(2.0)*refState[SimVar::dens]*(refState[SimVar::etot] - emin);
	C -= (refState[SimVar::momx]*refState[SimVar::momx] +
	      refState[SimVar::momy]*refState[SimVar::momy] +
	      refState[SimVar::momz]*refState[SimVar::momz]);

	Real rtdisc = sqrt(B*B - Real(4.0)*A*C);
	Real th1 = (-B + rtdisc)/(Real(2.0)*A),th2 = (-B - rtdisc)/(Real(2.0)*A);

	if (Real(0.0) <= th1 && th1 <= Real(1.0) && Real(0.0) <= th2 && th2 <= Real(1.0)) {
	  return th1 < th2 ? th1 : th2;
	} else if (Real(0.0) <= th1 && th1 <= Real(1.0)) {
	  return th1;
	} else if (Real(0.0) <= th2 && th2 <= Real(1.0)) {
	  return th2;
	} else {
	  return Real(0.0);
	}
      }
    };
    
  } // End namespace Hydro
  
} // End namespace KFVM

#endif
