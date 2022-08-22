#ifndef HYDRO_K_H_
#define HYDRO_K_H_

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdint>

#include "Definitions.H"

#include "../SimVar.H"
#include "../Types.H"
#include "EOS.H"
#include "Riemann.H"

namespace KFVM {

  namespace Hydro {

    template<class S>
    KOKKOS_INLINE_FUNCTION
    void Cons2Prim(S& state,const Real ref[NUM_VARS],Real gam)
    {
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

    template<class S>
    KOKKOS_INLINE_FUNCTION
    void Prim2Cons(S& state,const Real ref[NUM_VARS],Real gam)
    {
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

    struct RiemannSolverX_K
    {
      FaceDataView U;
      const Real gamma;

      RiemannSolverX_K(const FaceDataView& U_,Real gamma_):
	U(U_),gamma(gamma_) {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,Real& maxVel) const
      {
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
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
	  Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
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
      void operator()(const int i,const int j,const int k,Real& maxVel) const
      {
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
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
	  Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
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

    struct RiemannSolverY_K
    {
      FaceDataView U;
      const Real gamma;

      RiemannSolverY_K(const FaceDataView& U_,Real gamma_):
	U(U_),gamma(gamma_) {}

#if (SPACE_DIM == 2)
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,Real& maxVel) const
      {
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
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
	  Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
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
      void operator()(const int i,const int j,const int k,Real& maxVel) const
      {
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
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
	  Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
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

    struct RiemannSolverZ_K
    {
      FaceDataView U;
      const Real gamma;

      RiemannSolverZ_K(const FaceDataView& U_,Real gamma_):
	U(U_),gamma(gamma_) {}

#if (SPACE_DIM == 3)
      KOKKOS_INLINE_FUNCTION
      void operator()(const int i,const int j,const int k,Real& maxVel) const
      {
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
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
	  Real vq = EulerAdiabatic::HLL_Z(gamma,uL,uR,F);
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
    
  } // End namespace Hydro
  
} // End namespace KFVM

#endif
