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
#include "../SimVar.H"
#include "MHD_GLM_EOS.H"
#include "MHD_GLM_Riemann.H"
#include "Numeric.H"

namespace KFVM {

  namespace Physics {

    template<>
    class VarConvert<EquationType::MHD_GLM>
    {
      const Real gamm;
      const Real rho,u,v,w,bx,by,bz,rPsi,vsq;
      
    public:
      KOKKOS_INLINE_FUNCTION
      VarConvert(const MHD_GLM::ConsVec ref_,Real gam_):
	gamm(gam_ - Real(1.0)),
        rho(ref_[MHD_GLM::SimVar::dens]),
        u(ref_[MHD_GLM::SimVar::momx]/rho),
	v(ref_[MHD_GLM::SimVar::momy]/rho),
	w(ref_[MHD_GLM::SimVar::momz]/rho),
        bx(ref_[MHD_GLM::SimVar::magx]),
	by(ref_[MHD_GLM::SimVar::magy]),
	bz(ref_[MHD_GLM::SimVar::magz]),
	rPsi(ref_[MHD_GLM::SimVar::psi]),
	vsq((u*u + v*v + w*w)/Real(2.0))
      {}

      template<class T>
      KOKKOS_INLINE_FUNCTION
      void cons2LinPrim(T& state)
      {
        using MHD_GLM::SimVar;
        
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = gamm*(vsq*state(SimVar::dens) -
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

      template<class T>      
      KOKKOS_INLINE_FUNCTION
      void linPrim2Cons(T& state)
      {
        using MHD_GLM::SimVar;
        
        // Doing etot first allows state to be overwritten safely
        state(SimVar::etot) = (vsq*state(SimVar::dens) +
                               rho*u*state(SimVar::momx) + 
                               rho*v*state(SimVar::momy) + 
                               rho*w*state(SimVar::momz) +
                               bx*state(SimVar::magx) + 
                               by*state(SimVar::magy) + 
                               bz*state(SimVar::magz) +
                               state(SimVar::etot)/gamm +
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
        using MHD_GLM::ConsVec;
        using MHD_GLM::SimVar;
        using MHD_GLM::AuxVar;

	// Try not to assume the order of SimVars
	ConsVec u;
	for (int nV=0; nV<NUM_VARS; nV++) {
	  u[nV] = U(KFVM_D_DECL(i,j,k),nV);
	}
        
	// Set velocities
	V(KFVM_D_DECL(i,j,k),AuxVar::velx) = U(KFVM_D_DECL(i,j,k),SimVar::momx)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::vely) = U(KFVM_D_DECL(i,j,k),SimVar::momy)/U(KFVM_D_DECL(i,j,k),SimVar::dens);
	V(KFVM_D_DECL(i,j,k),AuxVar::velz) = U(KFVM_D_DECL(i,j,k),SimVar::momz)/U(KFVM_D_DECL(i,j,k),SimVar::dens);

	// Internal energy and pressure
	V(KFVM_D_DECL(i,j,k),AuxVar::eint) = EOS<EquationType::MHD_GLM>::eint(u);
	V(KFVM_D_DECL(i,j,k),AuxVar::pres) = EOS<EquationType::MHD_GLM>::pres(fProp.gamma,u);
	V(KFVM_D_DECL(i,j,k),AuxVar::prsg) = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,u);
	V(KFVM_D_DECL(i,j,k),AuxVar::prsb) = EOS<EquationType::MHD_GLM>::prsb(u);
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
        using MHD_GLM::ConsVec;
        using MHD_GLM::PrimVec;
        
        const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Pull out states and convert to primitive
	  ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
            uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uL,vL);
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uR,vR);
	  
	  // Calculate flux
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_X(fProp.gamma,fProp.ch_glm,uL,uR,vL,vR,F);
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
        using MHD_GLM::ConsVec;
        using MHD_GLM::PrimVec;
        
        const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // Pull out states and convert to primitive
	  ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
            uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uL,vL);
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uR,vR);
          
	  // Calculate flux
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_Y(fProp.gamma,fProp.ch_glm,uL,uR,vL,vR,F);
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
        using MHD_GLM::ConsVec;
        using MHD_GLM::PrimVec;
        
	for (idx_t nQ=0; nQ<NUM_QUAD_PTS*NUM_QUAD_PTS; nQ++) {
	  // Pull out states and convert to primitive
	  ConsVec uL,uR,F;
          PrimVec vL,vR;
	  for (int nV=0; nV<NUM_VARS; nV++) {
	    uL[nV] = U(KFVM_D_DECL(i,j,k),0,nQ,nV);
            uR[nV] = U(KFVM_D_DECL(i,j,k),1,nQ,nV);
	  }
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uL,vL);
          EOS<EquationType::MHD_GLM>::cons2prim(fProp.gamma,uR,vR);
          
	  // Calculate flux
	  Real vq = MHD_GLM_RS<rst>::RiemannSolve_Z(fProp.gamma,fProp.ch_glm,uL,uR,vL,vR,F);
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
	using Kokkos::fmax;
	using MHD_GLM::SimVar;
	using MHD_GLM::ConsVec;
	
	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;
	const Real pf = 0.99;

	// Check x-faces
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  // pull out states
	  Real cfVel;
	  ConsVec uM,uP;
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
	  ConsVec uM,uP;
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
	  ConsVec uM,uP;
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

    namespace MHD_GLM {
#include "SourceTerms.H"
    }

    template<>
    struct SourceTerms_K<EquationType::MHD_GLM>
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
	using MHD_GLM::SimVar;

        const int nzQuad = SPACE_DIM==2 ? 1 : NUM_QUAD_PTS;

        // Get cell center coordinate
        const Real xc = geom.xCoord(i);
        const Real yc = geom.yCoord(j);
#if (SPACE_DIM == 2)
        const Real zc = geom.zCoord(0);
#else
        const Real zc = geom.zCoord(k);
#endif

	// Set up temporaries for divergence and psi gradient
#if (SPACE_DIM == 2)
        Real divB[NUM_QUAD_PTS][NUM_QUAD_PTS][1];
        Real gpsiDu[NUM_QUAD_PTS][NUM_QUAD_PTS][1];
#else
        Real divB[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_QUAD_PTS];
        Real gpsiDu[NUM_QUAD_PTS][NUM_QUAD_PTS][NUM_QUAD_PTS];
#endif
	// get subviews of point values for this cell
	auto cellVals = Kokkos::subview(source,KFVM_D_DECL(i,j,k),KFVM_D_DECL(Kokkos::ALL,
									      Kokkos::ALL,
									      Kokkos::ALL),Kokkos::ALL);
	auto wFaceVals = Kokkos::subview(rsX,KFVM_D_DECL(i    ,j,k),1,Kokkos::ALL,Kokkos::ALL);
	auto eFaceVals = Kokkos::subview(rsX,KFVM_D_DECL(i + 1,j,k),0,Kokkos::ALL,Kokkos::ALL);
	auto sFaceVals = Kokkos::subview(rsY,KFVM_D_DECL(i,j    ,k),1,Kokkos::ALL,Kokkos::ALL);
	auto nFaceVals = Kokkos::subview(rsY,KFVM_D_DECL(i,j + 1,k),0,Kokkos::ALL,Kokkos::ALL);
#if (SPACE_DIM == 3)
	auto bFaceVals = Kokkos::subview(rsZ,KFVM_D_DECL(i,j,k    ),1,Kokkos::ALL,Kokkos::ALL);
	auto tFaceVals = Kokkos::subview(rsZ,KFVM_D_DECL(i,j,k + 1),0,Kokkos::ALL,Kokkos::ALL);
#endif
	
        // Find field divergence and psi gradient terms
        for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
          for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
            for (int nS=0; nS<nzQuad; nS++) {
              const Real rho = cellVals(KFVM_D_DECL(nQ,nR,nS),SimVar::dens);
              const Real vx = cellVals(KFVM_D_DECL(nQ,nR,nS),SimVar::momx)/rho;
              const Real vy = cellVals(KFVM_D_DECL(nQ,nR,nS),SimVar::momy)/rho;
              const Real vz = cellVals(KFVM_D_DECL(nQ,nR,nS),SimVar::momz)/rho;
	      
              divB[nQ][nR][nS] = diffX(cellVals,wFaceVals,eFaceVals,nQ,nR,nS,SimVar::magx);
              divB[nQ][nR][nS] += diffY(cellVals,sFaceVals,nFaceVals,nQ,nR,nS,SimVar::magy);

	      gpsiDu[nQ][nR][nS] = vx*diffX(cellVals,wFaceVals,eFaceVals,nQ,nR,nS,SimVar::psi);
	      gpsiDu[nQ][nR][nS] += vy*diffY(cellVals,sFaceVals,nFaceVals,nQ,nR,nS,SimVar::psi);
	      
	      // z-derivatives only if needed
#if (SPACE_DIM == 3)
              divB[nQ][nR][nS] += diffZ(cellVals,bFaceVals,tFaceVals,nQ,nR,nS,SimVar::magz);
	      gpsiDu[nQ][nR][nS] += vz*diffZ(cellVals,bFaceVals,tFaceVals,nQ,nR,nS,SimVar::psi);
#endif
            }
          }
        }

        // Find and fill forcing
        for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  Real x = Numeric::sclAb(xc,geom.dx,ab(nQ));
          for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	    Real y = Numeric::sclAb(yc,geom.dy,ab(nR));
            for (int nS=0; nS<nzQuad; nS++) {
	      Real z = SPACE_DIM == 2 ? zc : Numeric::sclAb(zc,geom.dz,ab(nS));
	      MHD_GLM::ConsVec state,force;
              // Extract state and clear force vector
              for (int nV=0; nV<NUM_VARS; nV++) {
                state[nV] = source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV);
                force[nV] = 0;
              }
              
              // Fill force with user defined terms
              MHD_GLM::SourceTerm(t,x,y,z,state,force);

              // Define GLM source terms
              const Real vx = state[SimVar::momx]/state[SimVar::dens];
              const Real vy = state[SimVar::momy]/state[SimVar::dens];
              const Real vz = state[SimVar::momz]/state[SimVar::dens];

              const Real uDb = vx*state[SimVar::magx] + vy*state[SimVar::magy] + vz*state[SimVar::magz];

              // Fill source view with GLM and user forcing
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::dens) =
                force[SimVar::dens];
              
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::momx) =
                force[SimVar::momx] - divB[nQ][nR][nS]*state[SimVar::magx];
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::momy) =
                force[SimVar::momy] - divB[nQ][nR][nS]*state[SimVar::magy];
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::momz) =
                force[SimVar::momz] - divB[nQ][nR][nS]*state[SimVar::magz];
	
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::etot) =
                force[SimVar::etot] - divB[nQ][nR][nS]*uDb + gpsiDu[nQ][nR][nS]*state[SimVar::psi];
	
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::magx) =
                force[SimVar::magx] - divB[nQ][nR][nS]*vx;
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::magy) =
                force[SimVar::magy] - divB[nQ][nR][nS]*vy;
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::magz) =
                force[SimVar::magz] - divB[nQ][nR][nS]*vz;
	
              source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),SimVar::psi) =
                force[SimVar::psi] - gpsiDu[nQ][nR][nS];
            }
          }
        }
      }

      template<class CellView,class MFaceView,class PFaceView>
      KOKKOS_INLINE_FUNCTION
      Real diffX(CellView& cellVals,const MFaceView& mFace,const PFaceView& pFace,
		 const int nQ,const int nR,const int nS,const int nV) const
      {
	// Evaluate boundary terms
	int faceIdx = SPACE_DIM == 2 ? nR : nR*NUM_QUAD_PTS + nS;
	Real dvdx = D(nQ,0)*mFace(faceIdx,nV) +
	  D(nQ,NUM_QUAD_PTS + 1)*pFace(faceIdx,nV);
	for (int n=1; n<=NUM_QUAD_PTS; n++) {
	  dvdx += D(nQ,n)*cellVals(KFVM_D_DECL(n - 1,nR,nS),nV);
	}
	dvdx /= geom.dx;
	return dvdx;
      }

      template<class CellView,class MFaceView,class PFaceView>
      KOKKOS_INLINE_FUNCTION
      Real diffY(CellView& cellVals,const MFaceView& mFace,const PFaceView& pFace,
		 const int nQ,const int nR,const int nS,const int nV) const
      {
	// Evaluate boundary terms
	int faceIdx = SPACE_DIM == 2 ? nQ : nQ*NUM_QUAD_PTS + nS;
	Real dvdy = D(nR,0)*mFace(faceIdx,nV) +
	  D(nR,NUM_QUAD_PTS + 1)*pFace(faceIdx,nV);
	for (int n=1; n<=NUM_QUAD_PTS; n++) {
	  dvdy += D(nR,n)*cellVals(KFVM_D_DECL(nQ,n - 1,nS),nV);
	}
	dvdy /= geom.dy;
	return dvdy;
      }

      template<class CellView,class MFaceView,class PFaceView>
      KOKKOS_INLINE_FUNCTION
      Real diffZ(CellView& cellVals,const MFaceView& mFace,const PFaceView& pFace,
		 const int nQ,const int nR,const int nS,const int nV) const
      {
	// Evaluate boundary terms
	int faceIdx = nQ*NUM_QUAD_PTS + nR;
	Real dvdz = D(nS,0)*mFace(faceIdx,nV) +
	  D(nS,NUM_QUAD_PTS + 1)*pFace(faceIdx,nV);
	for (int n=1; n<=NUM_QUAD_PTS; n++) {
	  dvdz += D(nS,n)*cellVals(KFVM_D_DECL(nQ,nR,n - 1),nV);
	}
	dvdz /= geom.dz;
	return dvdz;
      }
    };

    template<class UDataView>
    struct PositivityPreserve_K<EquationType::MHD_GLM,UDataView>
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
	using MHD_GLM::SimVar;
	using MHD_GLM::ConsVec;

	const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS*NUM_QUAD_PTS;

	// Save reference state
	ConsVec refState;
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
	      ConsVec state;
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
        Real dzW = Real(0.0);
#endif
        Real divV = (dxU + dyV + dzW)/Real(2.0); // Undivided difference
        Real kappa1 = Real(0.2),kappa2 = Real(0.2);
        Real eta = fmin(Real(1.0),fmax(Real(0.0),-(kappa1*minFms + divV)/(kappa1*minFms)));

        // Extend density and pressure ranges
        maxDens *= (Real(1.0) + kappa2 - kappa2*eta);
        minDens *= (Real(1.0) - kappa2 + kappa2*eta);
        minPres *= (Real(1.0) - kappa2 + kappa2*eta);
        
	Real rDensDiffP = maxDens - rDens,rDensDiffM = rDens - minDens;
	Real rPresDiff = rPres - minPres;

	// Enforce positive density
	// Find mixing amount given by face values
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
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		for (idx_t nV=0; nV<NUM_VARS - 1; nV++) {
		  source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) =
		    refState[nV] + theta*(source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) - refState[nV]);
		}
	      }
	    }
	  }
	}

	// Enforce positive pressure
	// Find mixing amount
	theta = Real(1.0);
	for (idx_t nQ=0; nQ<nQuad; nQ++) {
	  Real p,thEst;
          ConsVec state;
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
		Real p = EOS<EquationType::MHD_GLM>::prsg(fProp.gamma,state);
		Real thEst = rPresDiff/(rPres - p);
		theta = p >= minPres ? theta : fmin(theta,mhdPresPos(fProp.gamma,minPres,thEst,refState,state));
	      }
	    }
	  }
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
	if (haveSources) {
	  int nzQuad = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
	  for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	    for (int nR=0; nR<NUM_QUAD_PTS; nR++) {
	      for (int nS=0; nS<nzQuad; nS++) {
		for (idx_t nV=0; nV<NUM_VARS - 1; nV++) {
		  source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) =
		    refState[nV] + theta*(source(KFVM_D_DECL(i,j,k),KFVM_D_DECL(nQ,nR,nS),nV) - refState[nV]);
		}
	      }
	    }
	  }
	}
      }

      KOKKOS_INLINE_FUNCTION
      Real mhdPresPos(const Real gamma,const Real pmin,Real theta,
		      const MHD_GLM::ConsVec refState,const MHD_GLM::ConsVec state) const
      {
	using Kokkos::sqrt;
	using Kokkos::fmin;
	using MHD_GLM::SimVar;

	Real emin = pmin/(gamma - Real(1.0));

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
      
	Real B = Real(2.0)*(state[SimVar::dens] - refState[SimVar::dens])*(state[SimVar::etot] - refState[SimVar::etot]);
	B -= ((state[SimVar::momx] - refState[SimVar::momx])*(state[SimVar::momx] - refState[SimVar::momx]) +
	      (state[SimVar::momy] - refState[SimVar::momy])*(state[SimVar::momy] - refState[SimVar::momy]) +
	      (state[SimVar::momz] - refState[SimVar::momz])*(state[SimVar::momz] - refState[SimVar::momz]));
	B -= (refState[SimVar::dens]*bsq +
	      Real(2.0)*(state[SimVar::dens] - refState[SimVar::dens])*bdbr);
      
	Real C = Real(2.0)*(refState[SimVar::dens]*(state[SimVar::etot] - refState[SimVar::etot]) +
			    refState[SimVar::etot]*(state[SimVar::dens] - refState[SimVar::dens]));
	C -= Real(2.0)*((state[SimVar::momx] - refState[SimVar::momx])*refState[SimVar::momx] +
			(state[SimVar::momy] - refState[SimVar::momy])*refState[SimVar::momy] +
			(state[SimVar::momz] - refState[SimVar::momz])*refState[SimVar::momz]);
	C -= Real(2.0)*emin*(state[SimVar::dens] - refState[SimVar::dens]);
	C-= (Real(2.0)*refState[SimVar::dens]*bdbr + brsq*(state[SimVar::dens] - refState[SimVar::dens]));
      
	Real D = Real(2.0)*refState[SimVar::dens]*(refState[SimVar::etot] - emin) - refState[SimVar::dens]*bsq;
	D -= (refState[SimVar::momx]*refState[SimVar::momx] +
	      refState[SimVar::momy]*refState[SimVar::momy] +
	      refState[SimVar::momz]*refState[SimVar::momz]);

	for (int nIter=0; nIter<4; nIter++) {
	  theta -= (A*theta*theta*theta + B*theta*theta + C*theta + D)/(Real(3.0)*A*theta*theta + Real(2.0)*B*theta + C);
	}

	if (Real(0.0) <= theta && theta <= Real(1.0)) {
	  return theta;
	} else {
	  return Real(0.0);
	}
      }
    };
    
  } // End namespace MHD_GLM
  
} // End namespace KFVM

#endif
