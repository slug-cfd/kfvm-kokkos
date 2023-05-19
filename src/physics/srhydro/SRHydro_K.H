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
#include "../SimVar.H"
#include "SRHydro_EOS.H"
#include "SRHydro_Riemann.H"

#define KFVM_MAX_SR_ITER 200

namespace KFVM {

  namespace Physics {

    template<>
    class VarConvert<EquationType::SRHydro>
    {
      Real D,Sx,Sy,Sz,tau;
      
    public:
      
      KOKKOS_INLINE_FUNCTION
      VarConvert(const SRHydro::ConsVec ref_,Real gam_)
      {	
        using SRHydro::SimVar;
        (void) gam_;

	D = ref_[SimVar::dens];
	Sx = ref_[SimVar::momx];
	Sy = ref_[SimVar::momy];
	Sz = ref_[SimVar::momz];
	tau = ref_[SimVar::tau];
      }

      template<class T>
      KOKKOS_INLINE_FUNCTION
      void cons2LinPrim(T& state)
      {
        using SRHydro::SimVar;

        state(SimVar::momx) = state(SimVar::momx)/D - state(SimVar::dens)*Sx/(D*D);
        state(SimVar::momy) = state(SimVar::momy)/D - state(SimVar::dens)*Sy/(D*D);
        state(SimVar::momz) = state(SimVar::momz)/D - state(SimVar::dens)*Sz/(D*D);
      }

      template<class T>      
      KOKKOS_INLINE_FUNCTION
      void linPrim2Cons(T& state)
      {
        using SRHydro::SimVar;
        
        state(SimVar::momx) = state(SimVar::dens)*Sx/D + D*state(SimVar::momx);
        state(SimVar::momy) = state(SimVar::dens)*Sy/D + D*state(SimVar::momy);
        state(SimVar::momz) = state(SimVar::dens)*Sz/D + D*state(SimVar::momz);
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

    template<>
    struct SourceTerms_K<EquationType::SRHydro>
    {
      SourceDataView source;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      DiffMatView D;
      QuadRuleView ab;
      FluidProperties fProp;
      Geometry geom;
      Real t;

      SourceTerms_K(const SourceDataView source_,
                    KFVM_D_DECL(const FaceDataView& rsX_,
                                const FaceDataView& rsY_,
                                const FaceDataView& rsZ_),
                    const DiffMatView& D_,
		    const QuadRuleView& ab_,
		    const FluidProperties& fProp_,
		    const Geometry& geom_,
                    const Real t_):
	source(source_),
        KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
        D(D_),
	ab(ab_),
	fProp(fProp_),
	geom(geom_),
        t(t_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
	// Get coordinates of cell center
        Real xc = geom.xCoord(i);
        Real yc = geom.yCoord(j);
#if (SPACE_DIM == 2)
        Real zc = geom.zCoord(0);
#else
        Real zc = geom.zCoord(k);
#endif

        const int nzQuad = SPACE_DIM==2 ? 1 : NUM_QUAD_PTS;
        for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  Real x = Numeric::sclAb(xc,geom.dx,ab(nQ));
          for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    Real y = Numeric::sclAb(yc,geom.dy,ab(nR));
            for (int nS=0; nS<nzQuad; nS++) {
	      Real z = SPACE_DIM == 2 ? zc : Numeric::sclAb(zc,geom.dz,ab(nS));
	      SRHydro::ConsVec state,force;
              // Extract state and clear force vector
              for (int nV=0; nV<NUM_VARS; nV++) {
                state[nV] = source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV);
                force[nV] = 0;
              }
              
              // Fill force with user defined terms
              SRHydro::SourceTerm(t,x,y,z,state,force);

	      // overwrite source with forcing
	      for (int nV=0; nV<NUM_VARS; nV++) {
		source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) = force[nV];
	      }
            }
          }
        }
      }
    };

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::SRHydro,UDataView>
    {
      UDataView U;
      FaceDataView KFVM_D_DECL(rsX,rsY,rsZ);
      bool haveSources;
      SourceDataView source;
      FluidProperties fProp;

      PositivityPreserve_K(const UDataView& U_,
			   KFVM_D_DECL(const FaceDataView& rsX_,
				       const FaceDataView& rsY_,
				       const FaceDataView& rsZ_),
			   const bool haveSources_,
			   const SourceDataView& source_,
			   const FluidProperties& fProp_):
	U(U_),
	KFVM_D_DECL(rsX(rsX_),rsY(rsY_),rsZ(rsZ_)),
	haveSources(haveSources_),
	source(source_),
	fProp(fProp_)
      {}

      KOKKOS_INLINE_FUNCTION
      void operator()(KFVM_D_DECL(const idx_t i,const idx_t j,const idx_t k)) const
      {
	using Kokkos::fmin;
	using Kokkos::fmax;
	using SRHydro::SimVar;
	using SRHydro::ConsVec;

	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
        ConsVec refState;
	for (int nV=0; nV<NUM_VARS; nV++) {
	  refState[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
	Real rDens = refState[SimVar::dens];
	Real rTau = refState[SimVar::tau];
	
	// Find ranges of density and tau
	Real maxDens = -std::numeric_limits<Real>::max();
	Real minDens = std::numeric_limits<Real>::max();
	Real minTau = std::numeric_limits<Real>::max();
	Real minTauDiff = std::numeric_limits<Real>::max();
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
#if (SPACE_DIM == 3)
	    for (idx_t zo=-1; zo<=1; zo++) {
#endif
	      ConsVec state;
	      for (int nV=0; nV<NUM_VARS; nV++) {
		state[nV] = U(KFVM_D_DECL(i + xo,j + yo,k + zo),nV);
	      }
	      maxDens = fmax(maxDens,state[SimVar::dens]);
	      minDens = fmin(minDens,state[SimVar::dens]);
	      minTau = fmin(minTau,state[SimVar::tau]);
	      minTauDiff = fmin(minTauDiff,tauDiff(state));
#if (SPACE_DIM == 3)
	    }
#endif
	  }
	}

	// Enforce bounds on density
	Real rDensDiffP = maxDens - rDens,rDensDiffM = rDens - minDens;
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

	// Find mixing amount given by interior values
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		Real rho = source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::dens);
		theta = rho<minDens ? fmin(theta,rDensDiffM/(rDens - rho)) : theta;
		theta = rho>maxDens ? fmin(theta,rDensDiffP/(rho - rDens)) : theta;
	      }
	    }
	  }
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
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		for (idx_t nV=0; nV<NUM_VARS; nV++) {
		  source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) =
		    refState[nV] + theta*(source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) - refState[nV]);
		}
	      }
	    }
	  }
	}

	// Enforce lower bound on tau
	Real rTauDiff = rTau - minTau;
	// Find mixing amount
	theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // West face
          Real Tau = rsX(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
	  // East face
          Tau = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
	  // South face
          Tau = rsY(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
	  // North face
          Tau = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
#if (SPACE_DIM == 3)
	  // Bottom face
          Tau = rsZ(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
	  // Top face
          Tau = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::tau);
	  theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
#endif
	}

	// Find mixing amount given by interior values
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		Real Tau = source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::tau);
		theta = Tau<minTau ? fmin(theta,rTauDiff/(rTau - Tau)) : theta;
	      }
	    }
	  }
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
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		for (idx_t nV=0; nV<NUM_VARS; nV++) {
		  source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) =
		    refState[nV] + theta*(source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) - refState[nV]);
		}
	      }
	    }
	  }
	}

	// Enforce lower bound on tauDiff
	// Find mixing amount
	theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
          ConsVec state;
          Real td;
	  // West face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
	  // East face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
	  // South face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
	  // North face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
#if (SPACE_DIM == 3)
	  // Bottom face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
	  // Top face
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    state[nV] = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,nV);
	  }
          td = tauDiff(state);
          theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
#endif
	}

	// Find mixing amount given by interior values
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		ConsVec state;
		for (int nV=0; nV<NUM_VARS; nV++) {
		  state[nV] = source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV);
		}
		Real td = tauDiff(state);
		theta = td >= minTauDiff ? theta : fmin(theta,tauBound(minTauDiff,refState,state));
	      }
	    }
	  }
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
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		for (idx_t nV=0; nV<NUM_VARS; nV++) {
		  source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) =
		    refState[nV] + theta*(source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) - refState[nV]);
		}
	      }
	    }
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      Real tauDiff(const SRHydro::ConsVec state) const
      {
	using SRHydro::SimVar;
        
        const Real sSq = (state[SimVar::momx]*state[SimVar::momx] +
                          state[SimVar::momy]*state[SimVar::momy] +
                          state[SimVar::momz]*state[SimVar::momz]);
        
        return state[SimVar::tau]*state[SimVar::tau] + Real(2.0)*state[SimVar::dens]*state[SimVar::tau] - sSq;
      }
      
      KOKKOS_INLINE_FUNCTION
      Real tauBound(const Real minTauDiff,
                    const SRHydro::ConsVec refState,const SRHydro::ConsVec state) const
      {
	using Kokkos::sqrt;
	using SRHydro::SimVar;

        return Real(0.0);

        const Real sBarSq = (refState[SimVar::momx]*refState[SimVar::momx] +
                             refState[SimVar::momy]*refState[SimVar::momy] +
                             refState[SimVar::momz]*refState[SimVar::momz]);
        const Real sSq = (state[SimVar::momx]*state[SimVar::momx] +
                          state[SimVar::momy]*state[SimVar::momy] +
                          state[SimVar::momz]*state[SimVar::momz]);
        const Real sBarDS = (refState[SimVar::momx]*state[SimVar::momx] +
                             refState[SimVar::momy]*state[SimVar::momy] +
                             refState[SimVar::momz]*state[SimVar::momz]);
      
	const Real A = Real(2.0)*(refState[SimVar::dens]*refState[SimVar::tau] + state[SimVar::dens]*state[SimVar::tau] -
                                  refState[SimVar::dens]*state[SimVar::tau] - state[SimVar::dens]*refState[SimVar::tau] -
                                  refState[SimVar::tau]*state[SimVar::tau] +
                                  sBarDS) +
          refState[SimVar::tau]*refState[SimVar::tau] + state[SimVar::tau]*state[SimVar::tau] - sBarSq - sSq;
        
	const Real B = Real(2.0)*(refState[SimVar::dens]*state[SimVar::tau] + state[SimVar::dens]*refState[SimVar::tau] +
                                  refState[SimVar::tau]*state[SimVar::tau] + sBarSq - refState[SimVar::tau]*refState[SimVar::tau] -
                                  Real(2.0)*refState[SimVar::dens]*refState[SimVar::tau] - sBarDS);
      
	const Real C = (refState[SimVar::tau]*refState[SimVar::tau] +
                        Real(2.0)*refState[SimVar::dens]*refState[SimVar::tau] -
                        sBarSq - minTauDiff);

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
    
  } // End namespace SRHydro
  
} // End namespace KFVM

#endif
