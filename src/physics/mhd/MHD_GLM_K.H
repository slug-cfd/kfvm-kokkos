#ifndef MHD_GLM_K_H_
#define MHD_GLM_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <linux/limits.h>

#include "Definitions.H"

#include "../../Types.H"
#include "../EquationTypes.H"
#include "Dimension.H"
#include "Geometry.H"
#include "MHD_GLM_SimVar.H"
#include "MHD_GLM_EOS.H"
#include "../../FluidProperties.H"
#include "MHD_GLM_Riemann.H"
#include "physics/Physics_K.H"

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
	Real rPsi = ref[SimVar::psi];
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = gm*(vsq*state(SimVar::dens) -
                                  u*state(SimVar::momx) -
                                  v*state(SimVar::momy) -
                                  w*state(SimVar::momz) -
                                  bx*state(SimVar::magx) -
                                  by*state(SimVar::magy) -
                                  bz*state(SimVar::magz) +
                                  state(SimVar::etot) -
				  rPsi*state(SimVar::psi));
      
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
	Real rPsi = ref[SimVar::psi];
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = (vsq*state(SimVar::dens) +
                               rho*u*state(SimVar::momx) + 
                               rho*v*state(SimVar::momy) + 
                               rho*w*state(SimVar::momz) +
                               bx*state(SimVar::magx) + 
                               by*state(SimVar::magy) + 
                               bz*state(SimVar::magz) +
                               state(SimVar::etot)/gm +
			       rPsi*state(SimVar::psi));
      
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
	V(i,j,AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(ua);
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
	V(i,j,k,AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(ua);
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
	  Real vq = MHD_GLM_KEPES_X(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
	  Real vq = MHD_GLM_KEPES_X(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
	  Real vq = MHD_GLM_KEPES_Y(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
	  Real vq = MHD_GLM_KEPES_Y(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
	  Real vq = MHD_GLM_KEPES_Z(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
    struct SpeedEstimate_K<EquationType::MHD_GLM>
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
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k),Real& maxCh) const
      {
	using Kokkos::Experimental::fabs;
	using Kokkos::Experimental::fmax;
	using Kokkos::Experimental::fmin;
	using MHD_GLM::SimVar;
	
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	const Real pf = 0.99;

	// Check x-faces
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real cfVel;
	  Real uM[NUM_VARS],uP[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uM[nV] = rsX(KFVM_D_DECL(i    ,j,k),1,nQ,nV);
	    uP[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
	  
	  // Get fast mag sonic speed and longitudinal speed (x faces)
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicX(fProp.gamma,uM);
	  maxCh = fmax(maxCh,pf*cfVel);
	  
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicX(fProp.gamma,uP);
	  maxCh = fmax(maxCh,pf*cfVel);
	}

	// Check y-faces
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real cfVel;
	  Real uM[NUM_VARS],uP[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uM[nV] = rsY(KFVM_D_DECL(i,j    ,k),1,nQ,nV);
	    uP[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
	  
	  // Get fast mag sonic speed and longitudinal speed (y faces)
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicY(fProp.gamma,uM);
	  maxCh = fmax(maxCh,pf*cfVel);
	  
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicY(fProp.gamma,uP);
	  maxCh = fmax(maxCh,pf*cfVel);
	}

#if (SPACE_DIM == 3)
	// Check z-faces
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real cfVel;
	  Real uM[NUM_VARS],uP[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uM[nV] = rsZ(i,j,k    ,1,nQ,nV);
	    uP[nV] = rsZ(i,j,k + 1,0,nQ,nV);
	  }
	  
	  // Get fast mag sonic speed and longitudinal speed (z faces)
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicZ(fProp.gamma,uM);
	  maxCh = fmax(maxCh,pf*cfVel);
	  
	  cfVel = EOS<EquationType::MHD_GLM>::fastMagSonicZ(fProp.gamma,uP);
	  maxCh = fmax(maxCh,pf*cfVel);
	}
#endif
      }
    };

    template<class UDataView,class RHSDataView>
    struct SourceTerms_K<EquationType::MHD_GLM,UDataView,RHSDataView>
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
	using MHD_GLM::SimVar;
	
	const Real divB = 0.5*(U(i + 1,j,SimVar::magx) - U(i - 1,j,SimVar::magx))/geom.dx +
	  0.5*(U(i,j + 1,SimVar::magy) - U(i,j - 1,SimVar::magy))/geom.dy;

	const Real uDb = (U(i,j,SimVar::momx)*U(i,j,SimVar::magx) +
			  U(i,j,SimVar::momy)*U(i,j,SimVar::magy) +
			  U(i,j,SimVar::momz)*U(i,j,SimVar::magz))/U(i,j,SimVar::dens);

	const Real psi_x = 0.5*(U(i + 1,j,SimVar::psi) - U(i - 1,j,SimVar::psi))/geom.dx;
	const Real psi_y = 0.5*(U(i,j + 1,SimVar::psi) - U(i,j - 1,SimVar::psi))/geom.dy;

	const Real gpsiDu = (psi_x*U(i,j,SimVar::momx) + psi_y*U(i,j,SimVar::momy))/U(i,j,SimVar::dens);

	RHS(i,j,SimVar::momx) -= divB*U(i,j,SimVar::magx);
	RHS(i,j,SimVar::momy) -= divB*U(i,j,SimVar::magy);
	RHS(i,j,SimVar::momz) -= divB*U(i,j,SimVar::magz);
	
	RHS(i,j,SimVar::etot) -= divB*uDb + U(i,j,SimVar::psi)*gpsiDu;
	
	RHS(i,j,SimVar::magx) -= divB*U(i,j,SimVar::momx)/U(i,j,SimVar::dens);
	RHS(i,j,SimVar::magy) -= divB*U(i,j,SimVar::momy)/U(i,j,SimVar::dens);
	RHS(i,j,SimVar::magz) -= divB*U(i,j,SimVar::momz)/U(i,j,SimVar::dens);
	
	RHS(i,j,SimVar::psi) -= gpsiDu;
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const
      {
	using MHD_GLM::SimVar;
	
	const Real divB =
	  0.5*(U(i + 1,j,k,SimVar::magx) - U(i - 1,j,k,SimVar::magx))/geom.dx +
	  0.5*(U(i,j + 1,k,SimVar::magy) - U(i,j - 1,k,SimVar::magy))/geom.dy +
	  0.5*(U(i,j,k + 1,SimVar::magz) - U(i,j,k - 1,SimVar::magz))/geom.dz;

	const Real uDb = (U(i,j,k,SimVar::momx)*U(i,j,k,SimVar::magx) +
			  U(i,j,k,SimVar::momy)*U(i,j,k,SimVar::magy) +
			  U(i,j,k,SimVar::momz)*U(i,j,k,SimVar::magz))/U(i,j,k,SimVar::dens);

	const Real psi_x = 0.5*(U(i + 1,j,k,SimVar::psi) - U(i - 1,j,k,SimVar::psi))/geom.dx;
	const Real psi_y = 0.5*(U(i,j + 1,k,SimVar::psi) - U(i,j - 1,k,SimVar::psi))/geom.dy;
	const Real psi_z = 0.5*(U(i,j,k + 1,SimVar::psi) - U(i,j,k - 1,SimVar::psi))/geom.dz;

	const Real gpsiDu = (psi_x*U(i,j,k,SimVar::momx) +
			     psi_y*U(i,j,k,SimVar::momy) +
			     psi_z*U(i,j,k,SimVar::momz))/U(i,j,k,SimVar::dens);

	RHS(i,j,k,SimVar::momx) -= divB*U(i,j,k,SimVar::magx);
	RHS(i,j,k,SimVar::momy) -= divB*U(i,j,k,SimVar::magy);
	RHS(i,j,k,SimVar::momz) -= divB*U(i,j,k,SimVar::magz);
	
	RHS(i,j,k,SimVar::etot) -= divB*uDb + U(i,j,k,SimVar::psi)*gpsiDu;
	
	RHS(i,j,k,SimVar::magx) -= divB*U(i,j,k,SimVar::momx)/U(i,j,k,SimVar::dens);
	RHS(i,j,k,SimVar::magy) -= divB*U(i,j,k,SimVar::momy)/U(i,j,k,SimVar::dens);
	RHS(i,j,k,SimVar::magz) -= divB*U(i,j,k,SimVar::momz)/U(i,j,k,SimVar::dens);
	
	RHS(i,j,k,SimVar::psi) -= gpsiDu;
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

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
	using Kokkos::Experimental::fmin;
	using Kokkos::Experimental::fmax;
	using MHD_GLM::SimVar;

	constexpr idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
	Real refState[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13;
	Real minPres = 1.e-13;
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
#if (SPACE_DIM == 3)
	    for (idx_t zo=-1; zo<=1; zo++) {
#endif
	      Real state[NUM_VARS];
	      for (int nV=0; nV<NUM_VARS; nV++) {
		state[nV] = U(KFVM_D_DECL(i + xo,j + yo,k + zo),nV);
	      }
	      minDens = fmin(minDens,state[SimVar::dens]);
	      minPres = fmin(minPres,EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state));
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
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // East face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));
	  // South face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // North face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
#if (SPACE_DIM == 3)
	  // Bottom face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
	  theta = fmin(theta,fmax(1.0,rPresDiff/(rPres - p)));	  
	  // Top face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
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
    
  } // End namespace MHD_GLM
  
} // End namespace KFVM

#endif
