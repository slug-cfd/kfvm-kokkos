#ifndef MHD_GLM_K_H_
#define MHD_GLM_K_H_

#include <cstdint>

#include <Kokkos_Core.hpp>
#include <limits.h>

#include "Definitions.H"
#include "../../Dimension.H"
#include "../../Types.H"
#include "../../Geometry.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"
#include "../RSTypes.H"
#include "MHD_GLM_SimVar.H"
#include "MHD_GLM_EOS.H"
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

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
        using MHD_GLM::SimVar;
        using MHD_GLM::AuxVar;

	// Try not to assume the order of SimVars
	Real ua[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  ua[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
        
	// Set velocities
	V(KFVM_D_DECL(i,j,k),AuxVar::velx) = U(KFVM_D_DECL(i,j,k),SimVar::momx)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::vely) = U(KFVM_D_DECL(i,j,k),SimVar::momy)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::velz) = U(KFVM_D_DECL(i,j,k),SimVar::momz)/U(KFVM_D_DECL(i,j,k),SimVar::dens);

	// Internal energy and pressure
	V(KFVM_D_DECL(i,j,k),AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(ua);
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,ua);
	V(KFVM_D_DECL(i,j,k),AuxVar::prsg) = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,ua);
	V(KFVM_D_DECL(i,j,k),AuxVar::prsb) = EOS<EquationType::MHD_GLM>::prsb(ua);
      }
    };

    template<RSType rst>
    struct RiemannSolverX_K<EquationType::MHD_GLM,rst>
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
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
            uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  
	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_X(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
    struct RiemannSolverY_K<EquationType::MHD_GLM,rst>
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
	  Real uL[NUM_VARS],uR[NUM_VARS];
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  
	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_Y(fProp.gamma,fProp.ch_glm,uL,uR,F);
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
    struct RiemannSolverZ_K<EquationType::MHD_GLM,rst>
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
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_Z(fProp.gamma,fProp.ch_glm,uL,uR,F);
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

    KOKKOS_INLINE_FUNCTION
    Real mhdPresPos(const Real gamma,const Real pmin,Real theta,
                    const Real refState[NUM_VARS],const Real state[NUM_VARS])
    {
      using Kokkos::Experimental::sqrt;
      using Kokkos::Experimental::fmin;
      using MHD_GLM::SimVar;

      Real emin = pmin/(gamma - 1.0);

      Real bsq = (state[SimVar::magx]*state[SimVar::magx] +
                  state[SimVar::magy]*state[SimVar::magy] +
                  state[SimVar::magz]*state[SimVar::magz]);

      Real brsq = (refState[SimVar::magx]*refState[SimVar::magx] +
                   refState[SimVar::magy]*refState[SimVar::magy] +
                   refState[SimVar::magz]*refState[SimVar::magz]);

      Real bdbr = (state[SimVar::magx]*refState[SimVar::magx] +
                   state[SimVar::magy]*refState[SimVar::magy] +
                   state[SimVar::magz]*refState[SimVar::magz]);

      Real A = bsq*(refState[SimVar::dens] - state[SimVar::dens]);
      
      Real B = 2.0*(state[SimVar::dens] - refState[SimVar::dens])*(state[SimVar::etot] - refState[SimVar::etot]);
      B -= ((state[SimVar::momx] - refState[SimVar::momx])*(state[SimVar::momx] - refState[SimVar::momx]) +
            (state[SimVar::momy] - refState[SimVar::momy])*(state[SimVar::momy] - refState[SimVar::momy]) +
            (state[SimVar::momz] - refState[SimVar::momz])*(state[SimVar::momz] - refState[SimVar::momz]));
      B -= (refState[SimVar::dens]*bsq +
            2.0*(state[SimVar::dens] - refState[SimVar::dens])*bdbr);
      
      Real C = 2.0*(refState[SimVar::dens]*(state[SimVar::etot] - refState[SimVar::etot]) +
                    refState[SimVar::etot]*(state[SimVar::dens] - refState[SimVar::dens]));
      C -= 2.0*((state[SimVar::momx] - refState[SimVar::momx])*refState[SimVar::momx] +
                (state[SimVar::momy] - refState[SimVar::momy])*refState[SimVar::momy] +
                (state[SimVar::momz] - refState[SimVar::momz])*refState[SimVar::momz]);
      C -= 2.0*emin*(state[SimVar::dens] - refState[SimVar::dens]);
      C-= (2.0*refState[SimVar::dens]*bdbr + brsq*(state[SimVar::dens] - refState[SimVar::dens]));
      
      Real D = 2.0*refState[SimVar::dens]*(refState[SimVar::etot] - emin) - refState[SimVar::dens]*bsq;
      D -= (refState[SimVar::momx]*refState[SimVar::momx] +
            refState[SimVar::momy]*refState[SimVar::momy] +
            refState[SimVar::momz]*refState[SimVar::momz]);

      for (int nIter=0; nIter<4; nIter++) {
        theta -= (A*theta*theta*theta + B*theta*theta + C*theta + D)/(3.0*A*theta*theta + 2.0*B*theta + C);
      }

      if (0.0 <= theta && theta <= 1.0) {
        return theta;
      } else {
        return 0.0;
      }
    }

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

	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
	Real refState[NUM_VARS];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real rDens = refState[SimVar::dens];
	Real rPres = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,refState);
	
	// Find ranges of density, pressure, and soundspeed
	Real maxDens = -std::numeric_limits<Real>::max();
	Real minDens = std::numeric_limits<Real>::max();
	Real minPres = std::numeric_limits<Real>::max();
	Real minFms = std::numeric_limits<Real>::max();
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
              Real pres = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
              Real fms = EOS<EquationType::MHD_GLM>::soundspeed(fProp.gamma,state);
	      minPres = fmin(minPres,pres);
	      minFms = fmin(minFms,fms);
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
        Real dzW = 0.0;
#endif
        Real divV = (dxU + dyV + dzW)/2.0; // Undivided difference
        Real kappa1 = 0.2,kappa2 = 0.2;
        Real eta = fmin(1.0,fmax(0.0,-(kappa1*minFms + divV)/(kappa1*minFms)));

        // Extend density and pressure ranges
        maxDens *= (1.0 + kappa2 - kappa2*eta);
        minDens *= (1.0 - kappa2 + kappa2*eta);
        minPres *= (1.0 - kappa2 + kappa2*eta);
        
	Real rDensDiffP = maxDens - rDens,rDensDiffM = rDens - minDens;
	Real rPresDiff = rPres - minPres;

	// Enforce positive density
	// Find mixing amount
	Real theta = 1.0;
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
	  for (idx_t nV=0; nV<NUM_VARS - 1; nV++) {
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
	theta = 1.0;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  Real p,thEst,state[NUM_VARS];
	  // West face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
	  // East face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
	  // South face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
	  // North face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
#if (SPACE_DIM == 3)
	  // Bottom face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
	  // Top face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV);
	  }
	  p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
          thEst = rPresDiff/(rPres - p);
          theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
#endif
	}
	
	// Apply mixing
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  for (idx_t nV=0; nV<NUM_VARS - 1; nV++) {
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
