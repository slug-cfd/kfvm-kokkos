#ifndef MHD_GLM_K_H_
#define MHD_GLM_K_H_

#include <Kokkos_Macros.hpp>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "Definitions.H"

#include "../../Types.H"
#include "../EquationTypes.H"
#include "MHD_GLM_SimVar.H"
#include "MHD_GLM_EOS.H"
#include "../../FluidProperties.H"
#include "MHD_GLM_Riemann.H"

namespace KFVM {

  namespace Physics {

    template<class StateType>
    struct VarConvert<EquationType::MHD_GLM,StateType>
    {
      KOKKOS_INLINE_FUNCTION
      static void Cons2Prim(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using MHD_GLM::SimVar;
        
        Real gm = gam - 1.0;
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real bx = ref[SimVar::magx],by = ref[SimVar::magy],bz = ref[SimVar::magz];
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = gm*(vsq*state(SimVar::dens) -
                                  u*state(SimVar::momx) -
                                  v*state(SimVar::momy) -
                                  w*state(SimVar::momz) -
                                  bx*state(SimVar::magx) -
                                  by*state(SimVar::magy) -
                                  bz*state(SimVar::magz) +
                                  state(SimVar::etot));
      
        state(SimVar::momx) = (state(SimVar::momx) - u*state(SimVar::dens))/rho;
        state(SimVar::momy) = (state(SimVar::momy) - v*state(SimVar::dens))/rho;
        state(SimVar::momz) = (state(SimVar::momz) - w*state(SimVar::dens))/rho;
      }
      
      KOKKOS_INLINE_FUNCTION
      static void Prim2Cons(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using MHD_GLM::SimVar;
        
        Real gm = gam - 1.0;
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real bx = ref[SimVar::magx],by = ref[SimVar::magy],bz = ref[SimVar::magz];
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = (vsq*state(SimVar::dens) +
                               rho*u*state(SimVar::momx) + 
                               rho*v*state(SimVar::momy) + 
                               rho*w*state(SimVar::momz) +
                               bx*state(SimVar::magx) + 
                               by*state(SimVar::magy) + 
                               bz*state(SimVar::magz) +
                               state(SimVar::etot)/gm);
      
        state(SimVar::momx) = (rho*state(SimVar::momx) + u*state(SimVar::dens));
        state(SimVar::momy) = (rho*state(SimVar::momy) + v*state(SimVar::dens));
        state(SimVar::momz) = (rho*state(SimVar::momz) + w*state(SimVar::dens));
      }
    };

    template<class UViewType>
    struct AuxVars<EquationType::MHD_GLM,UViewType>
    {
      UViewType U;
      AuxDataView V;
      FluidProperties fProp;

      AuxVars(const UViewType& U_,const AuxDataView V_,const FluidProperties& fProp_):
	U(U_),
        V(V_),
        fProp(fProp_)
      {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const
      {
        using MHD_GLM::SimVar;
        using MHD_GLM::AuxVar;

	// Try not to assume the order of SimVars
	Real ua[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  ua[nV] = U(i,j,nV);
	}
        
	// Set velocities
	V(i,j,AuxVar::velx) = U(i,j,SimVar::momx)/U(i,j,SimVar::dens);
	V(i,j,AuxVar::vely) = U(i,j,SimVar::momy)/U(i,j,SimVar::dens);
	V(i,j,AuxVar::velz) = U(i,j,SimVar::momz)/U(i,j,SimVar::dens);

	// Internal energy and pressure
	V(i,j,AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(fProp.gamma,ua);
	V(i,j,AuxVar::pres) = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,ua);
	V(i,j,AuxVar::prsg) = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,ua);
	V(i,j,AuxVar::prsb) = EOS<EquationType::MHD_GLM>::prsb(ua);
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const
      {
        using MHD_GLM::SimVar;
        using MHD_GLM::AuxVar;

	// Try not to assume the order of SimVars
	Real ua[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  ua[nV] = U(i,j,k,nV);
	}
        
	// Set velocities
	V(i,j,k,AuxVar::velx) = U(i,j,k,SimVar::momx)/U(i,j,k,SimVar::dens);
	V(i,j,k,AuxVar::vely) = U(i,j,k,SimVar::momy)/U(i,j,k,SimVar::dens);
	V(i,j,k,AuxVar::velz) = U(i,j,k,SimVar::momz)/U(i,j,k,SimVar::dens);

	// Internal energy and pressure
	V(i,j,k,AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(fProp.gamma,ua);
	V(i,j,k,AuxVar::pres) = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,ua);
	V(i,j,k,AuxVar::prsg) = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,ua);
	V(i,j,k,AuxVar::prsb) = EOS<EquationType::MHD_GLM>::prsb(ua);
      }
#endif
    };

    template<>
    struct RiemannSolverX_K<EquationType::MHD_GLM>
    {
      FaceDataView U;
      FluidProperties fProp;

      RiemannSolverX_K(const FaceDataView& U_,const FluidProperties& fProp_):
	U(U_),
        fProp(fProp_)
      {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,0,nQ,nV);
	    uR[nV] = U(i,j,1,nQ,nV);
	  }

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_HLL_X(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(i,j,0,nQ,nV) = F[nV];
	    U(i,j,1,nQ,nV) = F[nV];
	  }
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,k,0,nQ,nV);
	    uR[nV] = U(i,j,k,1,nQ,nV);
	  }
	  
	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_HLL_X(fProp.gamma,uL,uR,F);
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
    struct RiemannSolverY_K<EquationType::MHD_GLM>
    {
      FaceDataView U;
      FluidProperties fProp;

      RiemannSolverY_K(const FaceDataView& U_,const FluidProperties& fProp_):
	U(U_),
        fProp(fProp_)
      {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,0,nQ,nV);
	    uR[nV] = U(i,j,1,nQ,nV);
	  }

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_HLL_Y(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    U(i,j,0,nQ,nV) = F[nV];
	    U(i,j,1,nQ,nV) = F[nV];
	  }
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,k,0,nQ,nV);
	    uR[nV] = U(i,j,k,1,nQ,nV);
	  }
	  
	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_HLL_Y(fProp.gamma,uL,uR,F);
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
    struct RiemannSolverZ_K<EquationType::MHD_GLM>
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
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(i,j,k,0,nQ,nV);
	    uR[nV] = U(i,j,k,1,nQ,nV);
	  }

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_HLL_Z(fProp.gamma,uL,uR,F);
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

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::MHD_GLM,UDataView>
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

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const
      {
	using Kokkos::Experimental::fmin;
        using MHD_GLM::SimVar;

	// Save reference state
	Real refState[] = {
	  U(i,j,SimVar::dens),
	  U(i,j,SimVar::momx),
	  U(i,j,SimVar::momy),
	  U(i,j,SimVar::momz),
	  U(i,j,SimVar::etot)
	}; 
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13,minPres = 1.e-13;
	
	// Adjust minima according to neighboring values
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
	    Real state[] = {
	      U(i + xo,j + yo,SimVar::dens),
	      U(i + xo,j + yo,SimVar::momx),
	      U(i + xo,j + yo,SimVar::momy),
	      U(i + xo,j + yo,SimVar::momz),
	      U(i + xo,j + yo,SimVar::etot)
	    };
	    minDens = fmin(minDens,state[0]);
	    minPres = fmin(minPres,EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state));
	  }
	}
	Real rDensDiff = rDens - minDens;
	Real rPresDiff = rPres - minPres;

	// Enforce positive density
	// Find mixing amount
	Real theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  Real th;
	  // West face
	  th = rsX(i,j,1,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i,j,1,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // East face
	  th = rsX(i + 1,j,0,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i + 1,j,0,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // South face
	  th = rsY(i,j,1,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j,1,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // North face
	  th = rsY(i,j + 1,0,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j + 1,0,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  rsX(i    ,j,1,nQ,SimVar::dens) = rDens + theta*(rsX(i    ,j,1,nQ,SimVar::dens) - rDens);
	  rsX(i + 1,j,0,nQ,SimVar::dens) = rDens + theta*(rsX(i + 1,j,0,nQ,SimVar::dens) - rDens);
	  rsY(i,j    ,1,nQ,SimVar::dens) = rDens + theta*(rsY(i,j    ,1,nQ,SimVar::dens) - rDens);
	  rsY(i,j + 1,0,nQ,SimVar::dens) = rDens + theta*(rsY(i,j + 1,0,nQ,SimVar::dens) - rDens);
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  Real th,p;
	  // West face
	  {
	    Real state[] = {
	      rsX(i,j,1,nQ,SimVar::dens),
	      rsX(i,j,1,nQ,SimVar::momx),
	      rsX(i,j,1,nQ,SimVar::momy),
	      rsX(i,j,1,nQ,SimVar::momz),
	      rsX(i,j,1,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // East face
	  {
	    Real state[] = {
	      rsX(i + 1,j,0,nQ,SimVar::dens),
	      rsX(i + 1,j,0,nQ,SimVar::momx),
	      rsX(i + 1,j,0,nQ,SimVar::momy),
	      rsX(i + 1,j,0,nQ,SimVar::momz),
	      rsX(i + 1,j,0,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // South face
	  {
	    Real state[] = {
	      rsY(i,j,1,nQ,SimVar::dens),
	      rsY(i,j,1,nQ,SimVar::momx),
	      rsY(i,j,1,nQ,SimVar::momy),
	      rsY(i,j,1,nQ,SimVar::momz),
	      rsY(i,j,1,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // North face
	  {
	    Real state[] = {
	      rsY(i,j + 1,0,nQ,SimVar::dens),
	      rsY(i,j + 1,0,nQ,SimVar::momx),
	      rsY(i,j + 1,0,nQ,SimVar::momy),
	      rsY(i,j + 1,0,nQ,SimVar::momz),
	      rsY(i,j + 1,0,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    rsX(i    ,j,1,nQ,nV) = refState[nV] + theta*(rsX(i    ,j,1,nQ,nV) - refState[nV]);
	    rsX(i + 1,j,0,nQ,nV) = refState[nV] + theta*(rsX(i + 1,j,0,nQ,nV) - refState[nV]);
	    rsY(i,j    ,1,nQ,nV) = refState[nV] + theta*(rsY(i,j    ,1,nQ,nV) - refState[nV]);
	    rsY(i,j + 1,0,nQ,nV) = refState[nV] + theta*(rsY(i,j + 1,0,nQ,nV) - refState[nV]);
	  }
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const
      {
	using Kokkos::Experimental::fmin;
        using MHD_GLM::SimVar;

	// Save reference state
	Real refState[] = {
	  U(i,j,k,SimVar::dens),
	  U(i,j,k,SimVar::momx),
	  U(i,j,k,SimVar::momy),
	  U(i,j,k,SimVar::momz),
	  U(i,j,k,SimVar::etot)
	}; 
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13,minPres = 1.e-13;
	
	// Adjust minima according to neighboring values
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
	    for (idx_t zo=-1; zo<=1; zo++) {
	      Real state[] = {
		U(i + xo,j + yo,k + zo,SimVar::dens),
		U(i + xo,j + yo,k + zo,SimVar::momx),
		U(i + xo,j + yo,k + zo,SimVar::momy),
		U(i + xo,j + yo,k + zo,SimVar::momz),
		U(i + xo,j + yo,k + zo,SimVar::etot)
	      };
	      minDens = fmin(minDens,state[0]);
	      minPres = fmin(minPres,EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state));
	    }
	  }
	}
	Real rDensDiff = rDens - minDens;
	Real rPresDiff = rPres - minPres;

	// Enforce positive density
	// Find mixing amount
	Real theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  Real th;
	  // West face
	  th = rsX(i,j,k,1,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i,j,k,1,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // East face
	  th = rsX(i + 1,j,k,0,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i + 1,j,k,0,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // South face
	  th = rsY(i,j,k,1,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j,k,1,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // North face
	  th = rsY(i,j + 1,k,0,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j + 1,k,0,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // Bottom face
	  th = rsZ(i,j,k,1,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsZ(i,j,k,1,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	  // Top face
	  th = rsZ(i,j,k + 1,0,nQ,SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsZ(i,j,k + 1,0,nQ,SimVar::dens));
	  theta = fmin(theta,th);
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  rsX(i    ,j,k,1,nQ,SimVar::dens) = rDens + theta*(rsX(i    ,j,k,1,nQ,SimVar::dens) - rDens);
	  rsX(i + 1,j,k,0,nQ,SimVar::dens) = rDens + theta*(rsX(i + 1,j,k,0,nQ,SimVar::dens) - rDens);
	  rsY(i,j    ,k,1,nQ,SimVar::dens) = rDens + theta*(rsY(i,j    ,k,1,nQ,SimVar::dens) - rDens);
	  rsY(i,j + 1,k,0,nQ,SimVar::dens) = rDens + theta*(rsY(i,j + 1,k,0,nQ,SimVar::dens) - rDens);
	  rsZ(i,j,k    ,1,nQ,SimVar::dens) = rDens + theta*(rsZ(i,j,k    ,1,nQ,SimVar::dens) - rDens);
	  rsZ(i,j,k + 1,0,nQ,SimVar::dens) = rDens + theta*(rsZ(i,j,k + 1,0,nQ,SimVar::dens) - rDens);
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  Real th,p;
	  // West face
	  {
	    Real state[] = {
	      rsX(i,j,k,1,nQ,SimVar::dens),
	      rsX(i,j,k,1,nQ,SimVar::momx),
	      rsX(i,j,k,1,nQ,SimVar::momy),
	      rsX(i,j,k,1,nQ,SimVar::momz),
	      rsX(i,j,k,1,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // East face
	  {
	    Real state[] = {
	      rsX(i + 1,j,k,0,nQ,SimVar::dens),
	      rsX(i + 1,j,k,0,nQ,SimVar::momx),
	      rsX(i + 1,j,k,0,nQ,SimVar::momy),
	      rsX(i + 1,j,k,0,nQ,SimVar::momz),
	      rsX(i + 1,j,k,0,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // South face
	  {
	    Real state[] = {
	      rsY(i,j,k,1,nQ,SimVar::dens),
	      rsY(i,j,k,1,nQ,SimVar::momx),
	      rsY(i,j,k,1,nQ,SimVar::momy),
	      rsY(i,j,k,1,nQ,SimVar::momz),
	      rsY(i,j,k,1,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // North face
	  {
	    Real state[] = {
	      rsY(i,j + 1,k,0,nQ,SimVar::dens),
	      rsY(i,j + 1,k,0,nQ,SimVar::momx),
	      rsY(i,j + 1,k,0,nQ,SimVar::momy),
	      rsY(i,j + 1,k,0,nQ,SimVar::momz),
	      rsY(i,j + 1,k,0,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // Bottom face
	  {
	    Real state[] = {
	      rsZ(i,j,k,1,nQ,SimVar::dens),
	      rsZ(i,j,k,1,nQ,SimVar::momx),
	      rsZ(i,j,k,1,nQ,SimVar::momy),
	      rsZ(i,j,k,1,nQ,SimVar::momz),
	      rsZ(i,j,k,1,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // Top face
	  {
	    Real state[] = {
	      rsZ(i,j,k + 1,0,nQ,SimVar::dens),
	      rsZ(i,j,k + 1,0,nQ,SimVar::momx),
	      rsZ(i,j,k + 1,0,nQ,SimVar::momy),
	      rsZ(i,j,k + 1,0,nQ,SimVar::momz),
	      rsZ(i,j,k + 1,0,nQ,SimVar::etot)
	    };
	    p = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS; nV++) {
	    rsX(i    ,j,k,1,nQ,nV) = refState[nV] + theta*(rsX(i    ,j,k,1,nQ,nV) - refState[nV]);
	    rsX(i + 1,j,k,0,nQ,nV) = refState[nV] + theta*(rsX(i + 1,j,k,0,nQ,nV) - refState[nV]);
	    rsY(i,j    ,k,1,nQ,nV) = refState[nV] + theta*(rsY(i,j    ,k,1,nQ,nV) - refState[nV]);
	    rsY(i,j + 1,k,0,nQ,nV) = refState[nV] + theta*(rsY(i,j + 1,k,0,nQ,nV) - refState[nV]);
	    rsZ(i,j,k    ,1,nQ,nV) = refState[nV] + theta*(rsZ(i,j,k    ,1,nQ,nV) - refState[nV]);
	    rsZ(i,j,k + 1,0,nQ,nV) = refState[nV] + theta*(rsZ(i,j,k + 1,0,nQ,nV) - refState[nV]);
	  }
	}
      }
#endif
    };
    
  } // End namespace Hydro
  
} // End namespace KFVM

#endif
