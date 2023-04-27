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
#include "physics/Physics_K.H"

#define KFVM_MAX_SR_ITER 200

namespace KFVM {

  namespace Physics {

    template<class StateType>
    class VarConvert<EquationType::SRHydro,StateType>
    {
      const Real gRatio;
      Real rho,W,Vx,Vy,Vz,p;
      Real D,Sx,Sy,Sz,tau;

      KOKKOS_INLINE_FUNCTION
      void fillP2C()
      {
	const Real W2 = W*W,W3 = W*W2,W4 = W2*W2,gWsq = gRatio*W2,rgp = rho + gRatio*p;
	const Real vsf = W2*rgp,vdf = 2*W4*rgp,tf = W3*(2*W*rgp - rho);
	
	p2c[0][0] = W;         p2c[0][1] = rho*W3*Vx;            p2c[0][2] = rho*W3*Vy;            p2c[0][3] = rho*W3*Vz;            p2c[0][4] = 0;
	p2c[1][0] = W2*Vx;     p2c[1][1] = vsf*(1 + 2*W2*Vx*Vx); p2c[1][2] = vdf*Vx*Vy;            p2c[1][3] = vdf*Vx*Vz;            p2c[1][4] = gWsq*Vx;
	p2c[2][0] = W2*Vy;     p2c[2][1] = vdf*Vy*Vx;            p2c[2][2] = vsf*(1 + 2*W2*Vy*Vy); p2c[2][3] = vdf*Vy*Vz;            p2c[2][4] = gWsq*Vy;
	p2c[3][0] = W2*Vz;     p2c[3][1] = vdf*Vz*Vx;            p2c[3][2] = vdf*Vz*Vy;            p2c[3][3] = vsf*(1 + 2*W2*Vz*Vz); p2c[3][4] = gWsq*Vz;
	p2c[4][0] = W*(W - 1); p2c[4][1] = tf*Vx;                p2c[4][2] = tf*Vy;                p2c[4][3] = tf*Vz;                p2c[4][4] = gWsq - 1;
      }

      // KOKKOS_INLINE_FUNCTION
      // void fillC2P()
      // {
      // 	// Intermediate quantites
      // 	const Real Wsq = W*W,rWsq = rho*Wsq,Wi = Real(1.0)/W;
      //   const Real h = Real(1.0) + gRatio*p/rho,hW = h*W;
      // 	const Real Ssq = Sx*Sx + Sy*Sy + Sz*Sz;
      // 	const Real A = D + tau + p,B = sqrt(A*A - Ssq),A3W4 = A*A*A*Wsq*Wsq;
      // 	const Real X = Real(1.0) + (D*Ssq*W + 2*gRatio*Ssq*p)/A3W4;
      // 	const Real Y = (D*W + 2*gRatio*p)/A3W4;
      //   const Real Ytil = Real(1.0) + Ssq*Y - gRatio/Wsq;

      // 	// Pressure derivatives
      // 	const Real dpdD = (Wi - X)/(X - gRatio/Wsq);
      // 	const Real dpdSx = Y*A*Sx/Ytil;
      // 	const Real dpdSy = Y*A*Sy/Ytil;
      // 	const Real dpdSz = Y*A*Sz/Ytil;
      // 	const Real dpdT = -X/(X - gRatio/Wsq);

      // 	// Lorenz derivatives
      // 	const Real dWidD = Ssq*(dpdD + Real(1.0))/(A*A*B);
      // 	const Real dWidSx = (Ssq*dpdSx - A*Sx)/(A*A*B);
      // 	const Real dWidSy = (Ssq*dpdSy - A*Sy)/(A*A*B);
      // 	const Real dWidSz = (Ssq*dpdSz - A*Sz)/(A*A*B);
      // 	const Real dWidT = Ssq*(dpdT + Real(1.0))/(A*A*B);

      // 	// Density derivatives
      // 	const Real drdD = Wi + D*dWidD;
      // 	const Real drdSx = D*dWidSx;
      // 	const Real drdSy = D*dWidSy;
      // 	const Real drdSz = D*dWidSz;
      // 	const Real drdT = D*dWidT;

      // 	// Enthalpy derivatives
      // 	const Real hidhdD = gRatio*(rho*dpdD - p*drdD)/(rho*rho*h);
      // 	const Real hidhdSx = gRatio*(rho*dpdSx - p*drdSx)/(rho*rho*h);
      // 	const Real hidhdSy = gRatio*(rho*dpdSy - p*drdSy)/(rho*rho*h);
      // 	const Real hidhdSz = gRatio*(rho*dpdSz - p*drdSz)/(rho*rho*h);
      // 	const Real hidhdT = gRatio*(rho*dpdT - p*drdT)/(rho*rho*h);

      // 	const Real dhdr = -gRatio*p/(rho*rho),dhdp = gRatio/rho;

      // 	// Velocity intermediates
      // 	const Real vd = Real(1.0)/D + hidhdD - dWidD/(W*Wsq);
      // 	const Real vs = Real(1.0)/(D*hW);
      // 	const Real vt = W*dWidT - hidhdT;
	
      // 	c2p[0][0] = drdD;   c2p[0][1] = drdSx;                        c2p[0][2] = drdSy;                        c2p[0][3] = drdSz;                        c2p[0][4] = drdT;
      // 	c2p[1][0] = -Vx*vd; c2p[1][1] = vs - Vx*(hidhdSx - dWidSx/(W*Wsq)); c2p[1][2] = -Vx*(hidhdSy - dWidSy/(W*Wsq));      c2p[1][3] = -Vx*(hidhdSz - dWidSz/(W*Wsq));      c2p[1][4] = Vx*vt;
      // 	c2p[2][0] = -Vy*vd; c2p[2][1] = -Vy*(hidhdSx - dWidSx/(W*Wsq));      c2p[2][2] = vs - Vy*(hidhdSy - dWidSy/(W*Wsq)); c2p[2][3] = -Vy*(hidhdSz - dWidSz/(W*Wsq));      c2p[2][4] = Vy*vt;
      // 	c2p[3][0] = -Vz*vd; c2p[3][1] = -Vz*(hidhdSx - dWidSx/(W*Wsq));      c2p[3][2] = -Vz*(hidhdSy - dWidSy/(W*Wsq));      c2p[3][3] = vs - Vz*(hidhdSz - dWidSz/(W*Wsq)); c2p[3][4] = Vz*vt;
      // 	c2p[4][0] = dpdD;   c2p[4][1] = dpdSx;                        c2p[4][2] = dpdSy;                        c2p[4][3] = dpdSz;                        c2p[4][4] = dpdT;
      // }

      KOKKOS_INLINE_FUNCTION
      void luInvert(const Real Ar[5][5],Real Ai[5][5])
      {
        // Copy Ar into temporary matrix, init Ai to eye
        Real A[5][5];
        for (int i=0; i<5; i++) {
          for (int j=0; j<5; j++) {
            A[i][j] = Ar[i][j];
            Ai[i][j] = i == j ? 1.0 : 0.0;
          }
        }

        // Forward sweep on A and Ai
        for (int j=0; j<4; j++) {
          for (int i=j+1; i<5; i++) {
            Real pf = A[i][j]/A[j][j];
            // Reduce A
            for (int k=0; k<5; k++) { A[i][k] -= pf*A[j][k]; }
            // Update Ai
            for (int k=0; k<5; k++) { Ai[i][k] -= pf*Ai[j][k]; }
          }
        }

        // Backward sweep
        for (int j=4; j>=0; j--) {
          for (int i=0; i<j; i++) {
            Real pf = A[i][j]/A[j][j];
            // Reduce A
            for (int k=0; k<5; k++) { A[i][k] -= pf*A[j][k]; }
            // Update Ai
            for (int k=0; k<5; k++) { Ai[i][k] -= pf*Ai[j][k]; }
          }
          // Scale row
          for (int k=0; k<5; k++) { Ai[j][k] /= A[j][j]; }
          A[j][j] = 1.0;
        }
      }
      
    public:
      // Real c2p[5][5];
      Real p2c[5][5];
      // Real c2pi[5][5];
      Real p2ci[5][5];
      
      KOKKOS_INLINE_FUNCTION
      VarConvert(const SRHydro::ConsVec ref_,Real gam_):
	gRatio(gam_/(gam_ - Real(1.0)))
      {
	using Kokkos::sqrt;
	
        using SRHydro::SimVar;
        using SRHydro::PrimVar;

	// Convert to primitive
	SRHydro::PrimVec primRef;
	EOS<EquationType::SRHydro>::cons2prim(gam_,KFVM_MAX_SR_ITER,ref_,primRef);
	
	// Unpack reference vectors
	rho = primRef[PrimVar::rho];
	W = primRef[PrimVar::lorz];
	Vx = primRef[PrimVar::velx];
	Vy = primRef[PrimVar::vely];
	Vz = primRef[PrimVar::velz];
	p = primRef[PrimVar::pres];
	
	D = ref_[SimVar::dens];
	Sx = ref_[SimVar::momx];
	Sy = ref_[SimVar::momy];
	Sz = ref_[SimVar::momz];
	tau = ref_[SimVar::tau];

	// Fill in Jacobians
	fillP2C();
	// fillC2P();

        luInvert(p2c,p2ci);
        // luInvert(c2p,c2pi);
      }
      
      KOKKOS_INLINE_FUNCTION
      void cons2LinPrim(StateType& state)
      {
        using SRHydro::SimVar;
	
        // J*state
	SRHydro::ConsVec lprim;
	for (int i=0; i<5; i++) {
	  lprim[i] = 0;
	  for (int j=0; j<5; j++) {
	    lprim[i] += p2ci[i][j]*state(j);
	  }
	}

        state(SimVar::dens) = lprim[SimVar::dens];
        state(SimVar::momx) = lprim[SimVar::momx];
        state(SimVar::momy) = lprim[SimVar::momy];
        state(SimVar::momz) = lprim[SimVar::momz];
        state(SimVar::tau) = lprim[SimVar::tau];
      }
      
      KOKKOS_INLINE_FUNCTION
      void linPrim2Cons(StateType& state)
      {
        using SRHydro::SimVar;
	
        // Jinv*state
	SRHydro::ConsVec cons;
	for (int i=0; i<5; i++) {
	  cons[i] = 0;
	  for (int j=0; j<5; j++) {
	    cons[i] += p2c[i][j]*state(j);
	  }
	}

        state(SimVar::dens) = cons[SimVar::dens];
        state(SimVar::momx) = cons[SimVar::momx];
        state(SimVar::momy) = cons[SimVar::momy];
        state(SimVar::momz) = cons[SimVar::momz];
        state(SimVar::tau) = cons[SimVar::tau];
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
	using Kokkos::sqrt;
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
	
	// Find ranges of density, pressure, and soundspeed
	Real maxDens = -std::numeric_limits<Real>::max();
	Real minDens = std::numeric_limits<Real>::max();
	Real minTau = std::numeric_limits<Real>::max();
	for (idx_t xo=-1; xo<=1; xo++) {
	  for (idx_t yo=-1; yo<=1; yo++) {
#if (SPACE_DIM == 3)
	    for (idx_t zo=-1; zo<=1; zo++) {
#endif
	      maxDens = fmax(maxDens,U(KFVM_D_DECL(i + xo,j + yo,k + zo),SimVar::dens));
	      minDens = fmin(minDens,U(KFVM_D_DECL(i + xo,j + yo,k + zo),SimVar::dens));
	      minTau = fmin(minTau,U(KFVM_D_DECL(i + xo,j + yo,k + zo),SimVar::tau));
#if (SPACE_DIM == 3)
	    }
#endif
	  }
	}

        // This hard floor on tau is overly restrictive, unsure how to remove though...
        minTau = fmax(minTau,1.0001*(sqrt(U(KFVM_D_DECL(i,j,k),SimVar::dens)*U(KFVM_D_DECL(i,j,k),SimVar::dens) +
					  U(KFVM_D_DECL(i,j,k),SimVar::momx)*U(KFVM_D_DECL(i,j,k),SimVar::momx) +
					  U(KFVM_D_DECL(i,j,k),SimVar::momy)*U(KFVM_D_DECL(i,j,k),SimVar::momy) +
					  U(KFVM_D_DECL(i,j,k),SimVar::momz)*U(KFVM_D_DECL(i,j,k),SimVar::momz)) -
				     U(KFVM_D_DECL(i,j,k),SimVar::dens)));
	
	// Enforce positive density
	Real rDensDiffP = maxDens - rDens,rDensDiffM = rDens - minDens;
	Real theta = Real(0.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // West face
          Real D = rsX(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
	  // East face
          D = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
	  // South face
          D = rsY(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
	  // North face
          D = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
#if (SPACE_DIM == 3)
	  // Bottom face
          D = rsZ(KFVM_D_DECL(i,j,k),1,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
	  // Top face
          D = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::dens);
	  theta = D<minDens ? fmin(theta,rDensDiffM/(rDens - D)) : theta;
	  theta = D>maxDens ? fmin(theta,rDensDiffP/(D - rDens)) : theta;
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
	
	// Enforce bounds on tau
	Real rTauDiffM = rTau - minTau;
	theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // West face
          Real tau = rsX(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
	  // East face
          tau = rsX(KFVM_D_DECL(i + 1,j,k),0,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
	  // South face
          tau = rsY(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
	  // North face
          tau = rsY(KFVM_D_DECL(i,j + 1,k),0,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
#if (SPACE_DIM == 3)
	  // Bottom face
          tau = rsZ(KFVM_D_DECL(i,j,k),1,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
	  // Top face
          tau = rsZ(KFVM_D_DECL(i,j,k + 1),0,nQ,SimVar::tau);
	  theta = tau < minTau ? fmin(theta,rTauDiffM/(rTau - tau)) : theta;
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
    
  } // End namespace SRHydro
  
} // End namespace KFVM

#endif
