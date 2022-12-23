#ifndef EULERADIABATIC_K_H_
#define EULERADIABATIC_K_H_

#include <Kokkos_Macros.hpp>
#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits>

#include "Definitions.H"

#include "../../Types.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"
#include "Dimension.H"
#include "EulerAdiabatic_SimVar.H"
#include "EulerAdiabatic_EOS.H"
#include "EulerAdiabatic_Riemann.H"
#include "Geometry.H"

namespace KFVM {

  namespace Physics {

    template<class StateType>
    struct VarConvert<EquationType::Euler,StateType>
    {
      KOKKOS_INLINE_FUNCTION
      static void Cons2Prim(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using EulerAdiabatic::SimVar;
	
        Real gm = gam - 1.0;
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/2.0;
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
        using EulerAdiabatic::SimVar;
	
        Real gm = gam - 1.0;
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/2.0;
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
    struct AuxVars<EquationType::Euler,UViewType>
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
        using EulerAdiabatic::SimVar;
        using EulerAdiabatic::AuxVar;

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
	V(KFVM_D_DECL(i,j,k),AuxVar::eint) = EOS<EquationType::Euler>::eint(ua);
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = EOS<EquationType::Euler>::pres(fProp.gamma,ua);
      }
    };

    template<>
    struct RiemannSolverX_K<EquationType::Euler>
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
	  Real vq = EulerHLL_X(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(KFVM_D_DECL(i,j,k),0,nQ,nV) = F[nV];
	    U(KFVM_D_DECL(i,j,k),1,nQ,nV) = F[nV];
	  }
	}
      }
    };

    template<>
    struct RiemannSolverY_K<EquationType::Euler>
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
	  Real vq = EulerHLL_Y(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	  
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(KFVM_D_DECL(i,j,k),0,nQ,nV) = F[nV];
	    U(KFVM_D_DECL(i,j,k),1,nQ,nV) = F[nV];
	  }
	}
      }
    };

    template<>
    struct RiemannSolverZ_K<EquationType::Euler>
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
	  Real vq = EulerHLL_Z(fProp.gamma,uL,uR,F);
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
    struct SpeedEstimate_K<EquationType::Euler>
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
      {
      }
    };

    template<class UDataView,class RHSDataView>
    struct SourceTerms_K<EquationType::Euler,UDataView,RHSDataView>
    {
      UDataView U;
      RHSDataView RHS;
      FluidProperties fProp;
      Geometry geom;

      SourceTerms_K(const UDataView& U_,const RHSDataView RHS_,
		    const FluidProperties& fProp_,const Geometry& geom_):
	U(U_),
	RHS(RHS_),
	fProp(fProp_),
	geom(geom_)
      {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const
      {
	return;
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const
      {
	return;
      }
#endif
    };

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::Euler,UDataView>
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
	using Kokkos::Experimental::fmin;
	using Kokkos::Experimental::fmax;
	using EulerAdiabatic::SimVar;

	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
	Real refState[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::Euler>::pres(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13;
	Real minPres = 1.e-13;
	for (idx_t xo=-2; xo<=2; xo++) {
	  for (idx_t yo=-2; yo<=2; yo++) {
#if (SPACE_DIM == 3)
	    for (idx_t zo=-2; zo<=2; zo++) {
#endif
	      Real state[NUM_VARS];
	      for (int nV=0; nV<NUM_VARS; nV++) {
		state[nV] = U(KFVM_D_DECL(i + xo,j + yo,k + zo),nV);
	      }
	      minDens = state[SimVar::dens] > 0 ? fmin(minDens,state[SimVar::dens]) : minDens;
              Real pres = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	      minPres = pres > 0 ? fmin(minPres,pres) : minPres;
#if (SPACE_DIM == 3)
	    }
#endif
	  }
	}
	Real rDensDiff = rDens - minDens;
	Real rPresDiff = rPres - minPres;

	// Enforce positive density
	// Find mixing amount
	Real theta = 1.0;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // West face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsX(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens))));
	  // East face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::dens))));
	  // South face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsY(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens))));
	  // North face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::dens))));
#if (SPACE_DIM == 3)
	  // Bottom face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsZ(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens))));
	  // Top face
	  theta = fmin(theta,fmax(1.0,
				  rDensDiff/(rDens - rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::dens))));
#endif
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  rsX(KFVM_D_DECL(i    ,j,k),1,nQ,SimVar::dens) = rDens + theta*(rsX(KFVM_D_DECL(i    ,j,k),1,nQ,SimVar::dens) - rDens);
	  rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::dens) = rDens + theta*(rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::dens) - rDens);
	  rsY(KFVM_D_DECL(i,j    ,k),1,nQ,SimVar::dens) = rDens + theta*(rsY(KFVM_D_DECL(i,j    ,k),1,nQ,SimVar::dens) - rDens);
	  rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::dens) = rDens + theta*(rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::dens) - rDens);
#if (SPACE_DIM == 3)
	  rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,SimVar::dens) = rDens + theta*(rsZ(KFVM_D_DECL(i,j,k    ),1,nQ,SimVar::dens) - rDens);
	  rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::dens) = rDens + theta*(rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::dens) - rDens);
#endif
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = 1.0;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  Real p,state[NUM_VARS];
	  // West face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // East face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));
	  // South face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // North face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
#if (SPACE_DIM == 3)
	  // Bottom face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // Top face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV);
	  }
	  p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));
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
    };
    
  } // End namespace Hydro
  
} // End namespace KFVM

#endif
