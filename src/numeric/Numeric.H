#ifndef NUMERIC_H_
#define NUMERIC_H_

// File: Numeric.H
// Purpose: Define reference data for various simple numerical methods
// Notes: The quadrature rules are all defined on (-0.5,0.5) with
//        a slightly hacky "/2.0" at the end of each entry

#include <cstdint>
#include <array>

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <impl/Kokkos_ViewCtor.hpp>

#include <Definitions.H>

#include "../Types.H"

namespace KFVM {

  namespace Numeric {

    // Helper function to convert abscissae on (-0.5,0.5)
    // to the bounds of a cell
    KOKKOS_INLINE_FUNCTION
    Real sclAb(const Real ctr,const Real del,const Real ab)
    {
      return ab*del + ctr;
    }

    // Lookup tables for specific values in Gaussian quadrature rules
    template<int nQuad> struct QuadRuleLUT
    {
      static std::array<Real,nQuad> ab,wt;
    };
    
    template <> std::array<Real,1> QuadRuleLUT<1>::ab;
    template <> std::array<Real,1> QuadRuleLUT<1>::wt;
    template <> std::array<Real,2> QuadRuleLUT<2>::ab;
    template <> std::array<Real,2> QuadRuleLUT<2>::wt;
    template <> std::array<Real,3> QuadRuleLUT<3>::ab;
    template <> std::array<Real,3> QuadRuleLUT<3>::wt;
    template <> std::array<Real,4> QuadRuleLUT<4>::ab;
    template <> std::array<Real,4> QuadRuleLUT<4>::wt;
    template <> std::array<Real,5> QuadRuleLUT<5>::ab;
    template <> std::array<Real,5> QuadRuleLUT<5>::wt;
    template <> std::array<Real,6> QuadRuleLUT<6>::ab;
    template <> std::array<Real,6> QuadRuleLUT<6>::wt;
    template <> std::array<Real,7> QuadRuleLUT<7>::ab;
    template <> std::array<Real,7> QuadRuleLUT<7>::wt;
    template <> std::array<Real,8> QuadRuleLUT<8>::ab;
    template <> std::array<Real,8> QuadRuleLUT<8>::wt;
    template <> std::array<Real,9> QuadRuleLUT<9>::ab;
    template <> std::array<Real,9> QuadRuleLUT<9>::wt;

    // Quadrature rule with values accessible in non-host space
    struct QuadRule
    {
      QuadRuleView ab,wt;
      QuadRule(): ab("QuadRuleAbscissae"),wt("QuadRuleWeights")
      {
	// Create host mirror and copy in from LUT
	auto h_ab = Kokkos::create_mirror(ab);
	auto h_wt = Kokkos::create_mirror(wt);

	for (int nQ=0; nQ<NUM_QUAD_PTS; nQ++) {
	  h_ab(nQ) = QuadRuleLUT<NUM_QUAD_PTS>::ab[nQ];
	  h_wt(nQ) = QuadRuleLUT<NUM_QUAD_PTS>::wt[nQ];
	}
	
	// Copy to views on the right memory space
	Kokkos::deep_copy(ab,h_ab);
	Kokkos::deep_copy(wt,h_wt);
      }
    };

    template<int nStage> struct SSPCoeff;
    
    // See Ruuth-Spiteri 2002
    // https://doi.org/10.1137/S0036142901389025
    // This is scheme SSP(5,4) given in appendix A
    template<> struct SSPCoeff<5>
    {
      // Coefficients on states
      constexpr static Real a_1_0 = 1.0;
      constexpr static Real a_2_0 = 0.44437049406734,a_2_1 = 0.55562950593266;
      constexpr static Real a_3_0 = 0.62010185138540,a_3_2 = 0.37989814861460;
      constexpr static Real a_4_0 = 0.17807995410773,a_4_3 = 0.82192004589227;
      constexpr static Real a_5_0 = 0.00683325884039,a_5_2 = 0.51723167208978,a_5_3 = 0.12759831133288,a_5_4 = 0.34833675773694;

      // Coefficients on RHS's
      constexpr static Real b_1 = 0.39175222700392;
      constexpr static Real b_2 = 0.36841059262959;
      constexpr static Real b_3 = 0.25189177424738;
      constexpr static Real b_4 = 0.54497475021237;
      constexpr static Real b_5 = 0.08460416338212,b_5til = 0.22600748319395;
      
    };
    
  } // End Numeric namespace
  
} // End KFVM namespace

#endif
