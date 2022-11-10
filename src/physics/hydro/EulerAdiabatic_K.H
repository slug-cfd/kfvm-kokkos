#ifndef EULERADIABATIC_K_H_
#define EULERADIABATIC_K_H_

#include <Kokkos_Macros.hpp>
#include <cstdint>

#include <Kokkos_Core.hpp>

#include "Definitions.H"

#include "../../Types.H"
#include "../../FluidProperties.H"
#include "../EquationTypes.H"
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
        Real gm = gam - 1.0;
        Real rho = ref[EulerAdiabatic::SimVar::dens];
        Real u = ref[EulerAdiabatic::SimVar::momx]/rho,v = ref[EulerAdiabatic::SimVar::momy]/rho,w = ref[EulerAdiabatic::SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(EulerAdiabatic::SimVar::etot) = gm*(vsq*state(EulerAdiabatic::SimVar::dens) -
                                  u*state(EulerAdiabatic::SimVar::momx) - 
                                  v*state(EulerAdiabatic::SimVar::momy) - 
                                  w*state(EulerAdiabatic::SimVar::momz) +
                                  state(EulerAdiabatic::SimVar::etot));
      
        state(EulerAdiabatic::SimVar::momx) = (state(EulerAdiabatic::SimVar::momx) - u*state(EulerAdiabatic::SimVar::dens))/rho;
        state(EulerAdiabatic::SimVar::momy) = (state(EulerAdiabatic::SimVar::momy) - v*state(EulerAdiabatic::SimVar::dens))/rho;
        state(EulerAdiabatic::SimVar::momz) = (state(EulerAdiabatic::SimVar::momz) - w*state(EulerAdiabatic::SimVar::dens))/rho;
      }
      
      KOKKOS_INLINE_FUNCTION
      static void Prim2Cons(StateType& state,const Real ref[NUM_VARS],Real gam)
      {
        Real gm = gam - 1.0;
        Real rho = ref[EulerAdiabatic::SimVar::dens];
        Real u = ref[EulerAdiabatic::SimVar::momx]/rho,v = ref[EulerAdiabatic::SimVar::momy]/rho,w = ref[EulerAdiabatic::SimVar::momz]/rho;
        Real vsq = (u*u + v*v + w*w)/2.0;
        // Doing etot first allows state to be overwritten safely
        state(EulerAdiabatic::SimVar::etot) = (vsq*state(EulerAdiabatic::SimVar::dens) +
                               rho*u*state(EulerAdiabatic::SimVar::momx) + 
                               rho*v*state(EulerAdiabatic::SimVar::momy) + 
                               rho*w*state(EulerAdiabatic::SimVar::momz) +
                               state(EulerAdiabatic::SimVar::etot)/gm);
      
        state(EulerAdiabatic::SimVar::momx) = (rho*state(EulerAdiabatic::SimVar::momx) + u*state(EulerAdiabatic::SimVar::dens));
        state(EulerAdiabatic::SimVar::momy) = (rho*state(EulerAdiabatic::SimVar::momy) + v*state(EulerAdiabatic::SimVar::dens));
        state(EulerAdiabatic::SimVar::momz) = (rho*state(EulerAdiabatic::SimVar::momz) + w*state(EulerAdiabatic::SimVar::dens));
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

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const
      {
	// Set velocities
	V(i,j,EulerAdiabatic::AuxVar::velx) = U(i,j,EulerAdiabatic::SimVar::momx)/U(i,j,EulerAdiabatic::SimVar::dens);
	V(i,j,EulerAdiabatic::AuxVar::vely) = U(i,j,EulerAdiabatic::SimVar::momy)/U(i,j,EulerAdiabatic::SimVar::dens);
	V(i,j,EulerAdiabatic::AuxVar::velz) = U(i,j,EulerAdiabatic::SimVar::momz)/U(i,j,EulerAdiabatic::SimVar::dens);

	// Internal energy and pressure
	Real ekin = U(i,j,EulerAdiabatic::SimVar::dens)*(V(i,j,EulerAdiabatic::AuxVar::velx)*V(i,j,EulerAdiabatic::AuxVar::velx) +
					 V(i,j,EulerAdiabatic::AuxVar::vely)*V(i,j,EulerAdiabatic::AuxVar::vely) +
					 V(i,j,EulerAdiabatic::AuxVar::velz)*V(i,j,EulerAdiabatic::AuxVar::velz))/2.0;
	V(i,j,EulerAdiabatic::AuxVar::eint) = (U(i,j,EulerAdiabatic::SimVar::etot) - ekin)/U(i,j,EulerAdiabatic::SimVar::dens);
	V(i,j,EulerAdiabatic::AuxVar::pres) = (1.0 - fProp.gamma)*(U(i,j,EulerAdiabatic::SimVar::etot) - ekin);
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k) const
      {
	// Set velocities
	V(i,j,k,EulerAdiabatic::AuxVar::velx) = U(i,j,k,EulerAdiabatic::SimVar::momx)/U(i,j,k,EulerAdiabatic::SimVar::dens);
	V(i,j,k,EulerAdiabatic::AuxVar::vely) = U(i,j,k,EulerAdiabatic::SimVar::momy)/U(i,j,k,EulerAdiabatic::SimVar::dens);
	V(i,j,k,EulerAdiabatic::AuxVar::velz) = U(i,j,k,EulerAdiabatic::SimVar::momz)/U(i,j,k,EulerAdiabatic::SimVar::dens);

	// Internal energy and pressure
	Real ekin = U(i,j,k,EulerAdiabatic::SimVar::dens)*(V(i,j,k,EulerAdiabatic::AuxVar::velx)*V(i,j,k,EulerAdiabatic::AuxVar::velx) +
					 V(i,j,k,EulerAdiabatic::AuxVar::vely)*V(i,j,k,EulerAdiabatic::AuxVar::vely) +
					 V(i,j,k,EulerAdiabatic::AuxVar::velz)*V(i,j,k,EulerAdiabatic::AuxVar::velz))/2.0;
	V(i,j,k,EulerAdiabatic::AuxVar::eint) = (U(i,j,k,EulerAdiabatic::SimVar::etot) - ekin)/U(i,j,k,EulerAdiabatic::SimVar::dens);
	V(i,j,k,EulerAdiabatic::AuxVar::pres) = (1.0 - fProp.gamma)*(U(i,j,k,EulerAdiabatic::SimVar::etot) - ekin);
      }
#endif
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

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  const Real uL[] = {U(i,j,0,nQ,0),
			     U(i,j,0,nQ,1),
			     U(i,j,0,nQ,2),
			     U(i,j,0,nQ,3),
			     U(i,j,0,nQ,4)};
	  const Real uR[] = {U(i,j,1,nQ,0),
			     U(i,j,1,nQ,1),
			     U(i,j,1,nQ,2),
			     U(i,j,1,nQ,3),
			     U(i,j,1,nQ,4)};

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = EulerHLL_X(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  U(i,j,0,nQ,0) = F[0]; U(i,j,1,nQ,0) = F[0];
	  U(i,j,0,nQ,1) = F[1]; U(i,j,1,nQ,1) = F[1];
	  U(i,j,0,nQ,2) = F[2]; U(i,j,1,nQ,2) = F[2];
	  U(i,j,0,nQ,3) = F[3]; U(i,j,1,nQ,3) = F[3];
	  U(i,j,0,nQ,4) = F[4]; U(i,j,1,nQ,4) = F[4];
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  const Real uL[] = {U(i,j,k,0,nQ,0),
			     U(i,j,k,0,nQ,1),
			     U(i,j,k,0,nQ,2),
			     U(i,j,k,0,nQ,3),
			     U(i,j,k,0,nQ,4)};
	  const Real uR[] = {U(i,j,k,1,nQ,0),
			     U(i,j,k,1,nQ,1),
			     U(i,j,k,1,nQ,2),
			     U(i,j,k,1,nQ,3),
			     U(i,j,k,1,nQ,4)};

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = EulerHLL_X(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  U(i,j,k,0,nQ,0) = F[0]; U(i,j,k,1,nQ,0) = F[0];
	  U(i,j,k,0,nQ,1) = F[1]; U(i,j,k,1,nQ,1) = F[1];
	  U(i,j,k,0,nQ,2) = F[2]; U(i,j,k,1,nQ,2) = F[2];
	  U(i,j,k,0,nQ,3) = F[3]; U(i,j,k,1,nQ,3) = F[3];
	  U(i,j,k,0,nQ,4) = F[4]; U(i,j,k,1,nQ,4) = F[4];
	}
      }
#endif
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

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  const Real uL[] = {U(i,j,0,nQ,0),
			     U(i,j,0,nQ,1),
			     U(i,j,0,nQ,2),
			     U(i,j,0,nQ,3),
			     U(i,j,0,nQ,4)};
	  const Real uR[] = {U(i,j,1,nQ,0),
			     U(i,j,1,nQ,1),
			     U(i,j,1,nQ,2),
			     U(i,j,1,nQ,3),
			     U(i,j,1,nQ,4)};

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = EulerHLL_Y(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  U(i,j,0,nQ,0) = F[0]; U(i,j,1,nQ,0) = F[0];
	  U(i,j,0,nQ,1) = F[1]; U(i,j,1,nQ,1) = F[1];
	  U(i,j,0,nQ,2) = F[2]; U(i,j,1,nQ,2) = F[2];
	  U(i,j,0,nQ,3) = F[3]; U(i,j,1,nQ,3) = F[3];
	  U(i,j,0,nQ,4) = F[4]; U(i,j,1,nQ,4) = F[4];
	}
      }
#else
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j,const idx_t k,Real& maxVel) const
      {
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // pull out states
	  const Real uL[] = {U(i,j,k,0,nQ,0),
			     U(i,j,k,0,nQ,1),
			     U(i,j,k,0,nQ,2),
			     U(i,j,k,0,nQ,3),
			     U(i,j,k,0,nQ,4)};
	  const Real uR[] = {U(i,j,k,1,nQ,0),
			     U(i,j,k,1,nQ,1),
			     U(i,j,k,1,nQ,2),
			     U(i,j,k,1,nQ,3),
			     U(i,j,k,1,nQ,4)};

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = EulerHLL_Y(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  U(i,j,k,0,nQ,0) = F[0]; U(i,j,k,1,nQ,0) = F[0];
	  U(i,j,k,0,nQ,1) = F[1]; U(i,j,k,1,nQ,1) = F[1];
	  U(i,j,k,0,nQ,2) = F[2]; U(i,j,k,1,nQ,2) = F[2];
	  U(i,j,k,0,nQ,3) = F[3]; U(i,j,k,1,nQ,3) = F[3];
	  U(i,j,k,0,nQ,4) = F[4]; U(i,j,k,1,nQ,4) = F[4];
	}
      }
#endif
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
	  const Real uL[] = {U(i,j,k,0,nQ,0),
			     U(i,j,k,0,nQ,1),
			     U(i,j,k,0,nQ,2),
			     U(i,j,k,0,nQ,3),
			     U(i,j,k,0,nQ,4)};
	  const Real uR[] = {U(i,j,k,1,nQ,0),
			     U(i,j,k,1,nQ,1),
			     U(i,j,k,1,nQ,2),
			     U(i,j,k,1,nQ,3),
			     U(i,j,k,1,nQ,4)};

	  // Calculate flux
	  Real F[NUM_VARS];
	  Real vq = EulerHLL_Z(fProp.gamma,uL,uR,F);
	  maxVel = vq > maxVel ? vq : maxVel;
	    
	  // Overwrite states with flux
	  U(i,j,k,0,nQ,0) = F[0]; U(i,j,k,1,nQ,0) = F[0];
	  U(i,j,k,0,nQ,1) = F[1]; U(i,j,k,1,nQ,1) = F[1];
	  U(i,j,k,0,nQ,2) = F[2]; U(i,j,k,1,nQ,2) = F[2];
	  U(i,j,k,0,nQ,3) = F[3]; U(i,j,k,1,nQ,3) = F[3];
	  U(i,j,k,0,nQ,4) = F[4]; U(i,j,k,1,nQ,4) = F[4];
	}
      }
#endif
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

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const idx_t i,const idx_t j) const
      {
	using Kokkos::Experimental::fmin;

	// Save reference state
	Real refState[] = {
	  U(i,j,EulerAdiabatic::SimVar::dens),
	  U(i,j,EulerAdiabatic::SimVar::momx),
	  U(i,j,EulerAdiabatic::SimVar::momy),
	  U(i,j,EulerAdiabatic::SimVar::momz),
	  U(i,j,EulerAdiabatic::SimVar::etot)
	}; 
	Real rDens = refState[EulerAdiabatic::SimVar::dens];
	Real rPres = EOS<EquationType::Euler>::pres(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13,minPres = 1.e-13;
	
	// Adjust minima according to neighboring values
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
	    Real state[] = {
	      U(i + xo,j + yo,EulerAdiabatic::SimVar::dens),
	      U(i + xo,j + yo,EulerAdiabatic::SimVar::momx),
	      U(i + xo,j + yo,EulerAdiabatic::SimVar::momy),
	      U(i + xo,j + yo,EulerAdiabatic::SimVar::momz),
	      U(i + xo,j + yo,EulerAdiabatic::SimVar::etot)
	    };
	    minDens = fmin(minDens,state[0]);
	    minPres = fmin(minPres,EOS<EquationType::Euler>::pres(fProp.gamma,state));
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
	  th = rsX(i,j,1,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i,j,1,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // East face
	  th = rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // South face
	  th = rsY(i,j,1,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j,1,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // North face
	  th = rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  rsX(i    ,j,1,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsX(i    ,j,1,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsY(i,j    ,1,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsY(i,j    ,1,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  Real th,p;
	  // West face
	  {
	    Real state[] = {
	      rsX(i,j,1,nQ,EulerAdiabatic::SimVar::dens),
	      rsX(i,j,1,nQ,EulerAdiabatic::SimVar::momx),
	      rsX(i,j,1,nQ,EulerAdiabatic::SimVar::momy),
	      rsX(i,j,1,nQ,EulerAdiabatic::SimVar::momz),
	      rsX(i,j,1,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // East face
	  {
	    Real state[] = {
	      rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::dens),
	      rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::momx),
	      rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::momy),
	      rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::momz),
	      rsX(i + 1,j,0,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // South face
	  {
	    Real state[] = {
	      rsY(i,j,1,nQ,EulerAdiabatic::SimVar::dens),
	      rsY(i,j,1,nQ,EulerAdiabatic::SimVar::momx),
	      rsY(i,j,1,nQ,EulerAdiabatic::SimVar::momy),
	      rsY(i,j,1,nQ,EulerAdiabatic::SimVar::momz),
	      rsY(i,j,1,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // North face
	  {
	    Real state[] = {
	      rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::dens),
	      rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::momx),
	      rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::momy),
	      rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::momz),
	      rsY(i,j + 1,0,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
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

	// Save reference state
	Real refState[] = {
	  U(i,j,k,EulerAdiabatic::SimVar::dens),
	  U(i,j,k,EulerAdiabatic::SimVar::momx),
	  U(i,j,k,EulerAdiabatic::SimVar::momy),
	  U(i,j,k,EulerAdiabatic::SimVar::momz),
	  U(i,j,k,EulerAdiabatic::SimVar::etot)
	}; 
	Real rDens = refState[EulerAdiabatic::SimVar::dens];
	Real rPres = EOS<EquationType::Euler>::pres(fProp.gamma,refState);
	
	// Minimum allowable density and pressure
	Real minDens = 1.e-13,minPres = 1.e-13;
	
	// Adjust minima according to neighboring values
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
	    for (idx_t zo=-1; zo<=1; zo++) {
	      Real state[] = {
		U(i + xo,j + yo,k + zo,EulerAdiabatic::SimVar::dens),
		U(i + xo,j + yo,k + zo,EulerAdiabatic::SimVar::momx),
		U(i + xo,j + yo,k + zo,EulerAdiabatic::SimVar::momy),
		U(i + xo,j + yo,k + zo,EulerAdiabatic::SimVar::momz),
		U(i + xo,j + yo,k + zo,EulerAdiabatic::SimVar::etot)
	      };
	      minDens = fmin(minDens,state[0]);
	      minPres = fmin(minPres,EOS<EquationType::Euler>::pres(fProp.gamma,state));
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
	  th = rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // East face
	  th = rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // South face
	  th = rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // North face
	  th = rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // Bottom face
	  th = rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	  // Top face
	  th = rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::dens) > minDens ? 1.0 :
	    rDensDiff/(rDens - rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::dens));
	  theta = fmin(theta,th);
	}

	// Apply mixing
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  rsX(i    ,j,k,1,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsX(i    ,j,k,1,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsY(i,j    ,k,1,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsY(i,j    ,k,1,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsZ(i,j,k    ,1,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsZ(i,j,k    ,1,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	  rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::dens) = rDens + theta*(rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::dens) - rDens);
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = 1.0;
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  Real th,p;
	  // West face
	  {
	    Real state[] = {
	      rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens),
	      rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::momx),
	      rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::momy),
	      rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::momz),
	      rsX(i,j,k,1,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // East face
	  {
	    Real state[] = {
	      rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::dens),
	      rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::momx),
	      rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::momy),
	      rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::momz),
	      rsX(i + 1,j,k,0,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // South face
	  {
	    Real state[] = {
	      rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens),
	      rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::momx),
	      rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::momy),
	      rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::momz),
	      rsY(i,j,k,1,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // North face
	  {
	    Real state[] = {
	      rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::dens),
	      rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::momx),
	      rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::momy),
	      rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::momz),
	      rsY(i,j + 1,k,0,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // Bottom face
	  {
	    Real state[] = {
	      rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::dens),
	      rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::momx),
	      rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::momy),
	      rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::momz),
	      rsZ(i,j,k,1,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
	    th = p > minPres ? 1.0 : rPresDiff/(rPres - p);
	    theta = fmin(theta,th);
	  }
	  // Top face
	  {
	    Real state[] = {
	      rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::dens),
	      rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::momx),
	      rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::momy),
	      rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::momz),
	      rsZ(i,j,k + 1,0,nQ,EulerAdiabatic::SimVar::etot)
	    };
	    p = EOS<EquationType::Euler>::pres(fProp.gamma,state);
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
