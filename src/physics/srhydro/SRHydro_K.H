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

#define KFVM_MAX_SR_ITER 10

namespace KFVM {

  namespace Physics {

    template<class StateType>
    class VarConvert<EquationType::SRHydro,StateType>
    {
      const Real gRatio;
      Real rho,W,Vx,Vy,Vz,p;
      Real D,Sx,Sy,Sz,tau;
      SRHydro::PrimVec primRef;
      
    public:
      Real J[5][5];
      Real Jinv[5][5];
      
      KOKKOS_INLINE_FUNCTION
      VarConvert(const SRHydro::ConsVec ref_,Real gam_):
	gRatio(gam_/(gam_ - Real(1.0)))
      {
	using Kokkos::sqrt;
	
        using SRHydro::SimVar;
        using SRHydro::PrimVar;

	// Convert to primitive
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

	// Intermediate quantites
	const Real Wsq = W*W,rWsq = rho*Wsq,h = Real(1.0) + gRatio*p/rho,hW = h*W;
	const Real Ssq = Sx*Sx + Sy*Sy + Sz*Sz;
	const Real A = D + tau + p,Bsq = A*A - Ssq,B = sqrt(Bsq);
	const Real X = 2*gRatio*p*Wsq*W*Ssq/(rho*A*A*A) + B*Ssq,Xtil = D*(gRatio*W/rho - X);
	const Real Y = 2*gRatio*W*p/(rho*Bsq) + Wsq*W*B*D/(rho*A*A*A);

	// Pressure derivatives
	const Real dpdD = (W - D*X - Real(1.0))/(Real(1.0) - Xtil);
	const Real dpdSx = D*Y*Sx/(Real(1.0) - Xtil);
	const Real dpdSy = D*Y*Sy/(Real(1.0) - Xtil);
	const Real dpdSz = D*Y*Sz/(Real(1.0) - Xtil);
	const Real dpdT = (D*X + Real(1.0))/(Xtil - Real(1.0));

	// Lorenz derivatives
	const Real dWidD = Ssq*(dpdD + Real(1.0))/(A*A*B);
	const Real dWidSx = (Ssq*dpdSx - A*Sx)/(A*A*B);
	const Real dWidSy = (Ssq*dpdSy - A*Sy)/(A*A*B);
	const Real dWidSz = (Ssq*dpdSz - A*Sz)/(A*A*B);
	const Real dWidT = Ssq*(dpdT + Real(1.0))/(A*A*B);

	// Density derivatives
	const Real drdD = Real(1.0)/W + D*dWidD;
	const Real drdSx = D*dWidSx;
	const Real drdSy = D*dWidSy;
	const Real drdSz = D*dWidSz;
	const Real drdT = D*dWidT;

	// Enthalpy derivatives
	const Real hidhdD = gRatio*(rho*dpdD - p*drdD)/(rho*rho*h);
	const Real hidhdSx = gRatio*(rho*dpdSx - p*drdSx)/(rho*rho*h);
	const Real hidhdSy = gRatio*(rho*dpdSy - p*drdSy)/(rho*rho*h);
	const Real hidhdSz = gRatio*(rho*dpdSz - p*drdSz)/(rho*rho*h);
	const Real hidhdT = gRatio*(rho*dpdT - p*drdT)/(rho*rho*h);

	const Real dhdr = -gRatio*p/(rho*rho),dhdp = gRatio/rho;

	// Velocity intermediates
	const Real vd = Real(1.0)/D + hidhdD + W*dWidD;
	const Real vs = Real(1.0)/(D*hW);
	const Real vt = W*dWidT - hidhdT;
	
	J[0][0] = drdD;   J[0][1] = drdSx;                        J[0][2] = drdSy;                        J[0][3] = drdSz;                        J[0][4] = drdT;
	J[1][0] = -Vx*vd; J[1][1] = vs + Vx*(W*dWidSx - hidhdSx); J[1][2] = Vx*(W*dWidSy - hidhdSy);      J[1][3] = Vx*(W*dWidSz - hidhdSz);      J[1][4] = Vx*vt;
	J[2][0] = -Vy*vd; J[2][1] = Vy*(W*dWidSx - hidhdSx);      J[2][2] = vs + Vy*(W*dWidSy - hidhdSy); J[2][3] = Vy*(W*dWidSz - hidhdSz);      J[2][4] = Vy*vt;
	J[3][0] = -Vz*vd; J[3][1] = Vz*(W*dWidSx - hidhdSx);      J[3][2] = Vz*(W*dWidSy - hidhdSy);      J[3][3] = vs + Vz*(W*dWidSz - hidhdSz); J[3][4] = Vz*vt;
	J[4][0] = dpdD;   J[4][1] = dpdSx;                        J[4][2] = dpdSy;                        J[4][3] = dpdSz;                        J[4][4] = dpdT;
	
        // Jacobian evaluated on reference
	const Real Dv = Wsq*(h + rho*dhdr);
	Jinv[0][0] = W;                           Jinv[0][1] = D*Wsq*Vx;                     Jinv[0][2] = D*Wsq*Vy;                      Jinv[0][3] = D*Wsq*Vz;                     Jinv[0][4] = 0.0;
	Jinv[1][0] = Vx*Dv;                       Jinv[1][1] = D*hW + 2*rho*h*Vx*Vx*Wsq*Wsq; Jinv[1][2] = 2*rho*h*Vx*Vy*Wsq*Wsq;         Jinv[1][3] = 2*rho*h*Vx*Vz*Wsq*Wsq;        Jinv[1][4] = rWsq*Vx*dhdp;
	Jinv[2][0] = Vy*Dv;                       Jinv[2][1] = 2*rho*h*Vy*Vx*Wsq*Wsq;        Jinv[2][2] = D*hW + 2*rho*h*Vy*Vy*Wsq*Wsq;  Jinv[2][3] = 2*rho*h*Vy*Vz*Wsq*Wsq;        Jinv[2][4] = rWsq*Vy*dhdp;
	Jinv[3][0] = Vz*Dv;                       Jinv[3][1] = 2*rho*h*Vz*Vx*Wsq*Wsq;        Jinv[3][2] = 2*rho*h*Vz*Vy*Wsq*Wsq;         Jinv[3][3] = D*hW + 2*rho*h*Vz*Vz*Wsq*Wsq; Jinv[3][4] = rWsq*Vz*dhdp;
	Jinv[4][0] = W*(D*dhdr + hW - Real(1.0)); Jinv[4][1] = 2*rho*h*Wsq*Wsq*Vx;           Jinv[4][2] = 2*rho*h*Wsq*Wsq*Vy;            Jinv[4][3] = 2*rho*h*Wsq*Wsq*Vz;           Jinv[4][4] = rWsq*dhdp - Real(1.0);
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
	    lprim[i] += J[i][j]*state(j);
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
	    cons[i] += Jinv[i][j]*state(j);
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
