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

#define KFVM_MAX_SR_ITER 10

namespace KFVM {

  namespace Physics {

    template<class StateType>
    struct VarConvert<EquationType::SRHydro,StateType>
    {
      // KOKKOS_INLINE_FUNCTION
      // static void cons2LinPrim(StateType& state,const SRHydro::ConsVec ref,Real gam)
      // {
      //   using SRHydro::SimVar;

      //   (void) gam;

      //   // Jacobian evaluated on ref, zeros omitted
      //   const Real dRdD = 1.0, dRdS1 = 0.0, dRdS2 = 0.0, dRdS3 = 0.0, dRdT = 0.0;
      //   const Real dV1dD = 0.0,dV1dS1 = 1.0,dV1dS2 = 0.0,dV1dS3 = 0.0,dV1dT = 0.0;
      //   const Real dV2dD = 0.0,dV2dS1 = 0.0,dV2dS2 = 1.0,dV2dS3 = 0.0,dV2dT = 0.0;
      //   const Real dV3dD = 0.0,dV3dS1 = 0.0,dV3dS2 = 0.0,dV3dS3 = 1.0,dV3dT = 0.0;
      //   const Real dPdD = 0.0, dPdS1 = 0.0, dPdS2 = 0.0, dPdS3 = 0.0, dPdT = 1.0;

      //   // J*state
      //   const Real R = dRdD*state(SimVar::dens) + dRdT*state(SimVar::tau) +
      //     dRdS1*state(SimVar::momx) + dRdS2*state(SimVar::momy) + dRdS3*state(SimVar::momz);
      //   const Real V1 = dV1dD*state(SimVar::dens) + dV1dT*state(SimVar::tau) +
      //     dV1dS1*state(SimVar::momx) + dV1dS2*state(SimVar::momy) + dV1dS3*state(SimVar::momz);
      //   const Real V2 = dV2dD*state(SimVar::dens) + dV2dT*state(SimVar::tau) +
      //     dV2dS1*state(SimVar::momx) + dV2dS2*state(SimVar::momy) + dV2dS3*state(SimVar::momz);
      //   const Real V3 = dV3dD*state(SimVar::dens) + dV3dT*state(SimVar::tau) +
      //     dV3dS1*state(SimVar::momx) + dV3dS2*state(SimVar::momy) + dV3dS3*state(SimVar::momz);
      //   const Real P = dPdD*state(SimVar::dens) + dPdT*state(SimVar::tau) +
      //     dPdS1*state(SimVar::momx) + dPdS2*state(SimVar::momy) + dPdS3*state(SimVar::momz);

      //   state(SimVar::dens) = R;
      //   state(SimVar::momx) = V1;
      //   state(SimVar::momy) = V2;
      //   state(SimVar::momz) = V3;
      //   state(SimVar::tau) = P;
      // }
      
      // KOKKOS_INLINE_FUNCTION
      // static void linPrim2Cons(StateType& state,const SRHydro::ConsVec ref,const Real gam)
      // {
      //   using SRHydro::SimVar;

      //   (void) gam;

      //   // Jacobian evaluated on ref, zeros omitted
      //   const Real dDdR = 1.0, dDdV1 = 0.0, dDdV2 = 0.0, dDdV3 = 0.0, dDdP = 0.0;
      //   const Real dS1dR = 0.0,dS1dV1 = 1.0,dS1dV2 = 0.0,dS1dV3 = 0.0,dS1dP = 0.0;
      //   const Real dS2dR = 0.0,dS2dV1 = 0.0,dS2dV2 = 1.0,dS2dV3 = 0.0,dS2dP = 0.0;
      //   const Real dS3dR = 0.0,dS3dV1 = 0.0,dS3dV2 = 0.0,dS3dV3 = 1.0,dS3dP = 0.0;
      //   const Real dTdR = 0.0, dTdV1 = 0.0, dTdV2 = 0.0, dTdV3 = 0.0, dTdP = 1.0;

      //   // J*state
      //   const Real D = dDdR*state(SimVar::dens) + dDdP*state(SimVar::tau) +
      //     dDdV1*state(SimVar::momx) + dDdV2*state(SimVar::momy) + dDdV3*state(SimVar::momz);
      //   const Real S1 = dS1dR*state(SimVar::dens) + dS1dP*state(SimVar::tau) +
      //     dS1dV1*state(SimVar::momx) + dS1dV2*state(SimVar::momy) + dS1dV3*state(SimVar::momz);
      //   const Real S2 = dS2dR*state(SimVar::dens) + dS2dP*state(SimVar::tau) +
      //     dS2dV1*state(SimVar::momx) + dS2dV2*state(SimVar::momy) + dS2dV3*state(SimVar::momz);
      //   const Real S3 = dS3dR*state(SimVar::dens) + dS3dP*state(SimVar::tau) +
      //     dS3dV1*state(SimVar::momx) + dS3dV2*state(SimVar::momy) + dS3dV3*state(SimVar::momz);
      //   const Real T = dTdR*state(SimVar::dens) + dTdP*state(SimVar::tau) +
      //     dTdV1*state(SimVar::momx) + dTdV2*state(SimVar::momy) + dTdV3*state(SimVar::momz);

      //   state(SimVar::dens) = D;
      //   state(SimVar::momx) = S1;
      //   state(SimVar::momy) = S2;
      //   state(SimVar::momz) = S3;
      //   state(SimVar::tau) = T;
      // }
      
      KOKKOS_INLINE_FUNCTION
      static void cons2LinPrim(StateType& state,const SRHydro::ConsVec ref,Real gam)
      {
        using SRHydro::SimVar;

        (void) gam;
        
        state(SimVar::dens) = ref[SimVar::dens];
        state(SimVar::momx) = ref[SimVar::momx];
        state(SimVar::momy) = ref[SimVar::momy];
        state(SimVar::momz) = ref[SimVar::momz];
        state(SimVar::tau) = ref[SimVar::tau];
      }
      
      KOKKOS_INLINE_FUNCTION
      static void linPrim2Cons(StateType& state,const SRHydro::ConsVec ref,const Real gam)
      {
        using SRHydro::SimVar;

        (void) gam;
        
        state(SimVar::dens) = ref[SimVar::dens];
        state(SimVar::momx) = ref[SimVar::momx];
        state(SimVar::momy) = ref[SimVar::momy];
        state(SimVar::momz) = ref[SimVar::momz];
        state(SimVar::tau) = ref[SimVar::tau];
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
        using SRHydro::PrimVar;
        using SRHydro::AuxVar;
        using SRHydro::ConsVec;
        using SRHydro::PrimVec;

	// Try not to assume the order of SimVars
        ConsVec u;
        PrimVec v;
	for (int nV=0; nV<NUM_VARS; nV++) {
	  u[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}

        // Call EOS for full primitive state all at once
        EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,u,v);
        
	// Copy everything back out
	V(KFVM_D_DECL(i,j,k),AuxVar::rho) = v[PrimVar::rho];
	V(KFVM_D_DECL(i,j,k),AuxVar::velx) = v[PrimVar::velx];
	V(KFVM_D_DECL(i,j,k),AuxVar::vely) = v[PrimVar::vely];
	V(KFVM_D_DECL(i,j,k),AuxVar::velz) = v[PrimVar::velz];
	V(KFVM_D_DECL(i,j,k),AuxVar::lorz) = v[PrimVar::lorz];
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = v[PrimVar::pres];
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
        using SRHydro::ConsVec;
        using SRHydro::PrimVec;
        
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Pull out states and convert to primitive
          ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uL,vL);
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uR,vR);

	  // Calculate flux
	  Real vq = SRHydroRS<rst>::RiemannSolve_X(fProp.gamma,uL,uR,vL,vR,F);
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
        using SRHydro::ConsVec;
        using SRHydro::PrimVec;
        
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Pull out states and convert to primitive
          ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uL,vL);
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uR,vR);

	  // Calculate flux
	  Real vq = SRHydroRS<rst>::RiemannSolve_Y(fProp.gamma,uL,uR,vL,vR,F);
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
        using SRHydro::ConsVec;
        using SRHydro::PrimVec;
        
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Pull out states and convert to primitive
          ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
	    uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uL,vL);
          EOS<EquationType::SRHydro>::cons2prim(fProp.gamma,KFVM_MAX_SR_ITER,uR,vR);
	  
	  // Calculate flux
	  Real vq = SRHydroRS<rst>::RiemannSolve_Z(fProp.gamma,uL,uR,vL,vR,F);
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
        SRHydro::ConsVec state,force;

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
	using Kokkos::fmin;
	using Kokkos::fmax;
	using SRHydro::SimVar;
      }

      KOKKOS_INLINE_FUNCTION
      Real eulerPresPos(const Real gamma,const Real pmin,
			const Real refState[NUM_VARS],const Real state[NUM_VARS]) const
      {
	using Kokkos::sqrt;
	using SRHydro::SimVar;
	return Real(0.0);
      }
    };
    
  } // End namespace SRHydro
  
} // End namespace KFVM

#endif
