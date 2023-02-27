#ifndef SRHYDRO_K_H_
#define SRHYDRO_K_H_

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
#include "SRHydro_SimVar.H"
#include "SRHydro_EOS.H"
#include "SRHydro_Riemann.H"

namespace KFVM {

  namespace Physics {

    template<class StateType>
    struct VarConvert<EquationType::SRHydro,StateType>
    {
      KOKKOS_INLINE_FUNCTION
      static void Cons2Prim(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using SRHydro::SimVar;
	
        Real gm = gam - Real(1.0);
        Real rho = ref[SimVar::dens];
        Real u = ref[SimVar::momx]/rho,v = ref[SimVar::momy]/rho,w = ref[SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/Real(2.0);
        // Doing tau first allows state to be overwritten safely
        state(SimVar::tau) = gm*(vsq*state(SimVar::dens) -
                                  u*state(SimVar::momx) - 
                                  v*state(SimVar::momy) - 
                                  w*state(SimVar::momz) +
                                  state(SimVar::tau));
      
        state(SimVar::momx) = (state(SimVar::momx) - u*state(SimVar::dens))/rho;
        state(SimVar::momy) = (state(SimVar::momy) - v*state(SimVar::dens))/rho;
        state(SimVar::momz) = (state(SimVar::momz) - w*state(SimVar::dens))/rho;
      }
      
      KOKKOS_INLINE_FUNCTION
      static void Prim2Cons(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        using SRHydro::SimVar;
      }
    };

    template<class UViewType>
    struct AuxVars<EquationType::SRHydro,UViewType>
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
        using SRHydro::SimVar;
        using SRHydro::AuxVar;

	// Try not to assume the order of SimVars
	Real ua[NUM_VARS],va[NUM_AUX];
	for (int nV=0; nV<NUM_VARS; nV++) {
	  ua[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}

        // Call EOS for full primitive state all at once
        EOS<EquationType::SRHydro>::primvec(fProp.gamma,ua,va);
        
	// Copy everything back out
	V(KFVM_D_DECL(i,j,k),AuxVar::rho) = va[AuxVar::rho];
	V(KFVM_D_DECL(i,j,k),AuxVar::lorenz) = va[AuxVar::lorenz];
	V(KFVM_D_DECL(i,j,k),AuxVar::velx) = va[AuxVar::velx];
	V(KFVM_D_DECL(i,j,k),AuxVar::vely) = va[AuxVar::vely];
	V(KFVM_D_DECL(i,j,k),AuxVar::velz) = va[AuxVar::velz];
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = va[AuxVar::pres];
      }
    };

    template<RSType rst>
    struct RiemannSolverX_K<EquationType::SRHydro,rst>
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
	  Real vq = SRHydroRS<rst>::RiemannSolve_X(fProp.gamma,uL,uR,F);
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
    struct RiemannSolverY_K<EquationType::SRHydro,rst>
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
	  Real vq = SRHydroRS<rst>::RiemannSolve_Y(fProp.gamma,uL,uR,F);
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
    struct RiemannSolverZ_K<EquationType::SRHydro,rst>
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
	  Real vq = SRHydroRS<rst>::RiemannSolve_Z(fProp.gamma,uL,uR,F);
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
    struct SpeedEstimate_K<EquationType::SRHydro>
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

    namespace SRHydro {
#include "SourceTerms.H"
    }

    template<class UDataView,class RHSDataView>
    struct SourceTerms_K<EquationType::SRHydro,UDataView,RHSDataView>
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
        SRHydro::SourceTerm(t,xc,yc,zc,state,force);

        // Apply to RHS
        for (int n=0; n<NUM_VARS; n++) {
          RHS(KFVM_D_DECL(i,j,k),n) += force[n];
        }
      }
    };

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::SRHydro,UDataView>
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
	using SRHydro::SimVar;
      }

      KOKKOS_INLINE_FUNCTION
      Real eulerPresPos(const Real gamma,const Real pmin,
			const Real refState[NUM_VARS],const Real state[NUM_VARS]) const
      {
	using Kokkos::Experimental::sqrt;
	using SRHydro::SimVar;
	return Real(0.0);
      }
    };
    
  } // End namespace SRHydro
  
} // End namespace KFVM

#endif
