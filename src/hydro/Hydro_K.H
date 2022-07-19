#ifndef HYDRO_K_H_
#define HYDRO_K_H_

#include <Kokkos_Macros.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdint>

#include <Definitions.H>
#include <Types.H>
#include <EOS.H>
#include <Riemann.H>

namespace KFVM {

  namespace Hydro {

    template<class RSViewType>
    struct RiemannSolver_K
    {
      // Work tags to select bulk or special cases near upper boundaries
      struct BulkTag{};
      struct EastTag{};
      struct NorthTag{};
      struct TopTag{};

      // View of Riemann states and fluid properties
      RSViewType U;
      const Real gamma;
      const int nX,nY,nZ;

      RiemannSolver_K(const RSViewType& U_,const ProblemSetup& ps):
	U(U_),
	gamma(1.4),
	nX(ps.nX),
	nY(ps.nY),
	nZ(ps.nZ) {}

      // Overloads on number of indices is ambiguous for reasons I can't figure out
      // This will get re-written anyway though
#if (SPACE_DIM == 2)
      // Compute 2d fluxes in bulk
      // This computes fluxes on the west and south faces
      KOKKOS_INLINE_FUNCTION
      void operator()(const BulkTag& tag,const int i,const int j,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  // West face
	  {
	    const Real uL[] = {U(i - 1,j,FaceLabel::east,nQ,0),
			       U(i - 1,j,FaceLabel::east,nQ,1),
			       U(i - 1,j,FaceLabel::east,nQ,2),
			       U(i - 1,j,FaceLabel::east,nQ,3),
			       U(i - 1,j,FaceLabel::east,nQ,4)};
	    const Real uR[] = {U(i,j,FaceLabel::west,nQ,0),
			       U(i,j,FaceLabel::west,nQ,1),
			       U(i,j,FaceLabel::west,nQ,2),
			       U(i,j,FaceLabel::west,nQ,3),
			       U(i,j,FaceLabel::west,nQ,4)};
	    Real F[NUM_VARS];
	    Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
	    v = vq>v ? vq : v;
	    
	    // Overwrite states with flux
	    U(i - 1,j,FaceLabel::east,nQ,0) = F[0];
	    U(i - 1,j,FaceLabel::east,nQ,1) = F[1];
	    U(i - 1,j,FaceLabel::east,nQ,2) = F[2];
	    U(i - 1,j,FaceLabel::east,nQ,3) = F[3];
	    U(i - 1,j,FaceLabel::east,nQ,4) = F[4];
	    
	    U(i,j,FaceLabel::west,nQ,0) = F[0];
	    U(i,j,FaceLabel::west,nQ,1) = F[1];
	    U(i,j,FaceLabel::west,nQ,2) = F[2];
	    U(i,j,FaceLabel::west,nQ,3) = F[3];
	    U(i,j,FaceLabel::west,nQ,4) = F[4];
	  }

	  // South face
	  {
	    const Real uL[] = {U(i,j - 1,FaceLabel::north,nQ,0),
			       U(i,j - 1,FaceLabel::north,nQ,1),
			       U(i,j - 1,FaceLabel::north,nQ,2),
			       U(i,j - 1,FaceLabel::north,nQ,3),
			       U(i,j - 1,FaceLabel::north,nQ,4)};
	    
	    const Real uR[] = {U(i,j,FaceLabel::south,nQ,0),
			       U(i,j,FaceLabel::south,nQ,1),
			       U(i,j,FaceLabel::south,nQ,2),
			       U(i,j,FaceLabel::south,nQ,3),
			       U(i,j,FaceLabel::south,nQ,4)};
	    Real F[NUM_VARS];
	    Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
	    v = vq>v ? vq : v;
	    
	    // Overwrite states with flux
	    U(i,j - 1,FaceLabel::north,nQ,0) = F[0];
	    U(i,j - 1,FaceLabel::north,nQ,1) = F[1];
	    U(i,j - 1,FaceLabel::north,nQ,2) = F[2];
	    U(i,j - 1,FaceLabel::north,nQ,3) = F[3];
	    U(i,j - 1,FaceLabel::north,nQ,4) = F[4];
	    
	    U(i,j,FaceLabel::south,nQ,0) = F[0];
	    U(i,j,FaceLabel::south,nQ,1) = F[1];
	    U(i,j,FaceLabel::south,nQ,2) = F[2];
	    U(i,j,FaceLabel::south,nQ,3) = F[3];
	    U(i,j,FaceLabel::south,nQ,4) = F[4];
	  }
	}
	maxVel = v>maxVel ? v : maxVel;
      }

      // Compute 2d flux on east boundary
      KOKKOS_INLINE_FUNCTION
      void operator()(const EastTag& tag,const int j,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  const Real uL[] = {U(nX - 1,j,FaceLabel::east,nQ,0),
			     U(nX - 1,j,FaceLabel::east,nQ,1),
			     U(nX - 1,j,FaceLabel::east,nQ,2),
			     U(nX - 1,j,FaceLabel::east,nQ,3),
			     U(nX - 1,j,FaceLabel::east,nQ,4)};
	  const Real uR[] = {U(nX,j,FaceLabel::west,nQ,0),
			     U(nX,j,FaceLabel::west,nQ,1),
			     U(nX,j,FaceLabel::west,nQ,2),
			     U(nX,j,FaceLabel::west,nQ,3),
			     U(nX,j,FaceLabel::west,nQ,4)};
	  Real F[NUM_VARS];
	  Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
	  v = vq>v ? vq : v;
	  
	  // Overwrite states with flux
	  U(nX - 1,j,FaceLabel::east,nQ,0) = F[0];
	  U(nX - 1,j,FaceLabel::east,nQ,1) = F[1];
	  U(nX - 1,j,FaceLabel::east,nQ,2) = F[2];
	  U(nX - 1,j,FaceLabel::east,nQ,3) = F[3];
	  U(nX - 1,j,FaceLabel::east,nQ,4) = F[4];
	    
	  U(nX,j,FaceLabel::west,nQ,0) = F[0];
	  U(nX,j,FaceLabel::west,nQ,1) = F[1];
	  U(nX,j,FaceLabel::west,nQ,2) = F[2];
	  U(nX,j,FaceLabel::west,nQ,3) = F[3];
	  U(nX,j,FaceLabel::west,nQ,4) = F[4];
	}
	maxVel = v>maxVel ? v : maxVel;
      }

      // Compute 2d fluxes on north boundary
      KOKKOS_INLINE_FUNCTION
      void operator()(const NorthTag& tag,const int i,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  const Real uL[] = {U(i,nY - 1,FaceLabel::north,nQ,0),
			     U(i,nY - 1,FaceLabel::north,nQ,1),
			     U(i,nY - 1,FaceLabel::north,nQ,2),
			     U(i,nY - 1,FaceLabel::north,nQ,3),
			     U(i,nY - 1,FaceLabel::north,nQ,4)};
	    
	  const Real uR[] = {U(i,nY,FaceLabel::south,nQ,0),
			     U(i,nY,FaceLabel::south,nQ,1),
			     U(i,nY,FaceLabel::south,nQ,2),
			     U(i,nY,FaceLabel::south,nQ,3),
			     U(i,nY,FaceLabel::south,nQ,4)};
	  Real F[NUM_VARS];
	  Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
	  v = vq>v ? vq : v;
	    
	  // Overwrite states with flux
	  U(i,nY - 1,FaceLabel::north,nQ,0) = F[0];
	  U(i,nY - 1,FaceLabel::north,nQ,1) = F[1];
	  U(i,nY - 1,FaceLabel::north,nQ,2) = F[2];
	  U(i,nY - 1,FaceLabel::north,nQ,3) = F[3];
	  U(i,nY - 1,FaceLabel::north,nQ,4) = F[4];
	    
	  U(i,nY,FaceLabel::south,nQ,0) = F[0];
	  U(i,nY,FaceLabel::south,nQ,1) = F[1];
	  U(i,nY,FaceLabel::south,nQ,2) = F[2];
	  U(i,nY,FaceLabel::south,nQ,3) = F[3];
	  U(i,nY,FaceLabel::south,nQ,4) = F[4];
	}
	maxVel = v>maxVel ? v : maxVel;
      }
#else
      // Compute 3d fluxes in bulk
      // This computes fluxes on the west, south, and bottom faces
      KOKKOS_INLINE_FUNCTION
      void operator()(const BulkTag& tag,const int i,const int j,const int k,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // West face
	  {
	    const Real uL[] = {U(i - 1,j,k,FaceLabel::east,nQ,0),
			       U(i - 1,j,k,FaceLabel::east,nQ,1),
			       U(i - 1,j,k,FaceLabel::east,nQ,2),
			       U(i - 1,j,k,FaceLabel::east,nQ,3),
			       U(i - 1,j,k,FaceLabel::east,nQ,4)};
	    const Real uR[] = {U(i,j,k,FaceLabel::west,nQ,0),
			       U(i,j,k,FaceLabel::west,nQ,1),
			       U(i,j,k,FaceLabel::west,nQ,2),
			       U(i,j,k,FaceLabel::west,nQ,3),
			       U(i,j,k,FaceLabel::west,nQ,4)};
	    Real F[NUM_VARS];
	    Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
	    v = vq>v ? vq : v;
	      
	    // Overwrite states with flux
	    U(i - 1,j,k,FaceLabel::east,nQ,0) = F[0];
	    U(i - 1,j,k,FaceLabel::east,nQ,1) = F[1];
	    U(i - 1,j,k,FaceLabel::east,nQ,2) = F[2];
	    U(i - 1,j,k,FaceLabel::east,nQ,3) = F[3];
	    U(i - 1,j,k,FaceLabel::east,nQ,4) = F[4];
	      
	    U(i,j,k,FaceLabel::west,nQ,0) = F[0];
	    U(i,j,k,FaceLabel::west,nQ,1) = F[1];
	    U(i,j,k,FaceLabel::west,nQ,2) = F[2];
	    U(i,j,k,FaceLabel::west,nQ,3) = F[3];
	    U(i,j,k,FaceLabel::west,nQ,4) = F[4];
	  }

	  // South face
	  {
	    const Real uL[] = {U(i,j - 1,k,FaceLabel::north,nQ,0),
			       U(i,j - 1,k,FaceLabel::north,nQ,1),
			       U(i,j - 1,k,FaceLabel::north,nQ,2),
			       U(i,j - 1,k,FaceLabel::north,nQ,3),
			       U(i,j - 1,k,FaceLabel::north,nQ,4)};
	    
	    const Real uR[] = {U(i,j,k,FaceLabel::south,nQ,0),
			       U(i,j,k,FaceLabel::south,nQ,1),
			       U(i,j,k,FaceLabel::south,nQ,2),
			       U(i,j,k,FaceLabel::south,nQ,3),
			       U(i,j,k,FaceLabel::south,nQ,4)};
	    Real F[NUM_VARS];
	    Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
	    v = vq>v ? vq : v;
	    
	    // Overwrite states with flux
	    U(i,j - 1,k,FaceLabel::north,nQ,0) = F[0];
	    U(i,j - 1,k,FaceLabel::north,nQ,1) = F[1];
	    U(i,j - 1,k,FaceLabel::north,nQ,2) = F[2];
	    U(i,j - 1,k,FaceLabel::north,nQ,3) = F[3];
	    U(i,j - 1,k,FaceLabel::north,nQ,4) = F[4];
	    
	    U(i,j,k,FaceLabel::south,nQ,0) = F[0];
	    U(i,j,k,FaceLabel::south,nQ,1) = F[1];
	    U(i,j,k,FaceLabel::south,nQ,2) = F[2];
	    U(i,j,k,FaceLabel::south,nQ,3) = F[3];
	    U(i,j,k,FaceLabel::south,nQ,4) = F[4];
	  }

	  // bottom face
	  {
	    const Real uL[] = {U(i,j,k - 1,FaceLabel::top,nQ,0),
			       U(i,j,k - 1,FaceLabel::top,nQ,1),
			       U(i,j,k - 1,FaceLabel::top,nQ,2),
			       U(i,j,k - 1,FaceLabel::top,nQ,3),
			       U(i,j,k - 1,FaceLabel::top,nQ,4)};
	    
	    const Real uR[] = {U(i,j,k,FaceLabel::bottom,nQ,0),
			       U(i,j,k,FaceLabel::bottom,nQ,1),
			       U(i,j,k,FaceLabel::bottom,nQ,2),
			       U(i,j,k,FaceLabel::bottom,nQ,3),
			       U(i,j,k,FaceLabel::bottom,nQ,4)};
	    Real F[NUM_VARS];
	    Real vq = EulerAdiabatic::HLL_Z(gamma,uL,uR,F);
	    v = vq>v ? vq : v;
	    
	    // Overwrite states with flux
	    U(i,j,k - 1,FaceLabel::top,nQ,0) = F[0];
	    U(i,j,k - 1,FaceLabel::top,nQ,1) = F[1];
	    U(i,j,k - 1,FaceLabel::top,nQ,2) = F[2];
	    U(i,j,k - 1,FaceLabel::top,nQ,3) = F[3];
	    U(i,j,k - 1,FaceLabel::top,nQ,4) = F[4];
	    
	    U(i,j,k,FaceLabel::bottom,nQ,0) = F[0];
	    U(i,j,k,FaceLabel::bottom,nQ,1) = F[1];
	    U(i,j,k,FaceLabel::bottom,nQ,2) = F[2];
	    U(i,j,k,FaceLabel::bottom,nQ,3) = F[3];
	    U(i,j,k,FaceLabel::bottom,nQ,4) = F[4];
	  }
	}
	maxVel = v>maxVel ? v : maxVel;
      }

      // Compute 3d flux on east boundary
      KOKKOS_INLINE_FUNCTION
      void operator()(const EastTag& tag,const int j,const int k,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  const Real uL[] = {U(nX - 1,j,k,FaceLabel::east,nQ,0),
			     U(nX - 1,j,k,FaceLabel::east,nQ,1),
			     U(nX - 1,j,k,FaceLabel::east,nQ,2),
			     U(nX - 1,j,k,FaceLabel::east,nQ,3),
			     U(nX - 1,j,k,FaceLabel::east,nQ,4)};
	  const Real uR[] = {U(nX,j,k,FaceLabel::west,nQ,0),
			     U(nX,j,k,FaceLabel::west,nQ,1),
			     U(nX,j,k,FaceLabel::west,nQ,2),
			     U(nX,j,k,FaceLabel::west,nQ,3),
			     U(nX,j,k,FaceLabel::west,nQ,4)};
	  Real F[NUM_VARS];
	  Real vq = EulerAdiabatic::HLL_X(gamma,uL,uR,F);
	  v = vq>v ? vq : v;
	  
	  // Overwrite states with flux
	  U(nX - 1,j,k,FaceLabel::east,nQ,0) = F[0];
	  U(nX - 1,j,k,FaceLabel::east,nQ,1) = F[1];
	  U(nX - 1,j,k,FaceLabel::east,nQ,2) = F[2];
	  U(nX - 1,j,k,FaceLabel::east,nQ,3) = F[3];
	  U(nX - 1,j,k,FaceLabel::east,nQ,4) = F[4];
	    
	  U(nX,j,k,FaceLabel::west,nQ,0) = F[0];
	  U(nX,j,k,FaceLabel::west,nQ,1) = F[1];
	  U(nX,j,k,FaceLabel::west,nQ,2) = F[2];
	  U(nX,j,k,FaceLabel::west,nQ,3) = F[3];
	  U(nX,j,k,FaceLabel::west,nQ,4) = F[4];
	}
	maxVel = v>maxVel ? v : maxVel;
      }

      // Compute 3d fluxes on north boundary
      KOKKOS_INLINE_FUNCTION
      void operator()(const NorthTag& tag,const int i,const int k,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  const Real uL[] = {U(i,nY - 1,k,FaceLabel::north,nQ,0),
			     U(i,nY - 1,k,FaceLabel::north,nQ,1),
			     U(i,nY - 1,k,FaceLabel::north,nQ,2),
			     U(i,nY - 1,k,FaceLabel::north,nQ,3),
			     U(i,nY - 1,k,FaceLabel::north,nQ,4)};
	    
	  const Real uR[] = {U(i,nY,k,FaceLabel::south,nQ,0),
			     U(i,nY,k,FaceLabel::south,nQ,1),
			     U(i,nY,k,FaceLabel::south,nQ,2),
			     U(i,nY,k,FaceLabel::south,nQ,3),
			     U(i,nY,k,FaceLabel::south,nQ,4)};
	  Real F[NUM_VARS];
	  Real vq = EulerAdiabatic::HLL_Y(gamma,uL,uR,F);
	  v = vq>v ? vq : v;
	    
	  // Overwrite states with flux
	  U(i,nY - 1,k,FaceLabel::north,nQ,0) = F[0];
	  U(i,nY - 1,k,FaceLabel::north,nQ,1) = F[1];
	  U(i,nY - 1,k,FaceLabel::north,nQ,2) = F[2];
	  U(i,nY - 1,k,FaceLabel::north,nQ,3) = F[3];
	  U(i,nY - 1,k,FaceLabel::north,nQ,4) = F[4];
	    
	  U(i,nY,k,FaceLabel::south,nQ,0) = F[0];
	  U(i,nY,k,FaceLabel::south,nQ,1) = F[1];
	  U(i,nY,k,FaceLabel::south,nQ,2) = F[2];
	  U(i,nY,k,FaceLabel::south,nQ,3) = F[3];
	  U(i,nY,k,FaceLabel::south,nQ,4) = F[4];
	}
	maxVel = v>maxVel ? v : maxVel;
      }

      // Compute 3d fluxes on top boundary
      KOKKOS_INLINE_FUNCTION
      void operator()(const TopTag& tag,const int i,const int j,Real& maxVel) const {
	(void) tag;
	// Loop over all quadrature points, track maximum velocity
	Real v = 0.0;
	for (int nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  const Real uL[] = {U(i,j,nZ - 1,FaceLabel::top,nQ,0),
			     U(i,j,nZ - 1,FaceLabel::top,nQ,1),
			     U(i,j,nZ - 1,FaceLabel::top,nQ,2),
			     U(i,j,nZ - 1,FaceLabel::top,nQ,3),
			     U(i,j,nZ - 1,FaceLabel::top,nQ,4)};
	    
	  const Real uR[] = {U(i,j,nZ,FaceLabel::bottom,nQ,0),
			     U(i,j,nZ,FaceLabel::bottom,nQ,1),
			     U(i,j,nZ,FaceLabel::bottom,nQ,2),
			     U(i,j,nZ,FaceLabel::bottom,nQ,3),
			     U(i,j,nZ,FaceLabel::bottom,nQ,4)};
	  Real F[NUM_VARS];
	  Real vq = EulerAdiabatic::HLL_Z(gamma,uL,uR,F);
	  v = vq>v ? vq : v;
	    
	  // Overwrite states with flux
	  U(i,j,nZ - 1,FaceLabel::top,nQ,0) = F[0];
	  U(i,j,nZ - 1,FaceLabel::top,nQ,1) = F[1];
	  U(i,j,nZ - 1,FaceLabel::top,nQ,2) = F[2];
	  U(i,j,nZ - 1,FaceLabel::top,nQ,3) = F[3];
	  U(i,j,nZ - 1,FaceLabel::top,nQ,4) = F[4];
	    
	  U(i,j,nZ,FaceLabel::bottom,nQ,0) = F[0];
	  U(i,j,nZ,FaceLabel::bottom,nQ,1) = F[1];
	  U(i,j,nZ,FaceLabel::bottom,nQ,2) = F[2];
	  U(i,j,nZ,FaceLabel::bottom,nQ,3) = F[3];
	  U(i,j,nZ,FaceLabel::bottom,nQ,4) = F[4];
	}
	maxVel = v>maxVel ? v : maxVel;
      }
#endif
    };
    
  } // End namespace Hydro
  
} // End namespace KFVM

#endif
