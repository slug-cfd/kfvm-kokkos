#ifndef NUMERIC_H_
#define NUMERIC_H_

// File: Numeric.H
// Purpose: Define reference data for various simple numerical methods
// Notes: The quadrature rules are all defined on (-0.5,0.5) with
//        a slightly hacky "/2.0" at the end of each entry

#include <Kokkos_CopyViews.hpp>
#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_Layout.hpp>
#include <cstdint>
#include <array>
#include <impl/Kokkos_ViewCtor.hpp>

#include "Types.H"

namespace KFVM {

  namespace Numeric {

    // Lookup tables for specific values in Gaussian quadrature rules
    template<int nQuad> struct QuadRuleLUT
    {
      static std::array<Real,nQuad> ab,wt;
    };

    // Quadrature rule with values accessible in non-host space
    template<int nQuad> struct QuadRule
    {
      Kokkos::View<Real[nQuad]> ab,wt;
      QuadRule(): ab("QuadRuleAbscissae"),wt("QuadRuleWeights")
      {
	// Create host mirror and copy in from LUT
	auto h_ab = Kokkos::create_mirror_view(ab);
	auto h_wt = Kokkos::create_mirror_view(wt);

	for (int nQ=0; nQ<nQuad; nQ++) {
	  h_ab(nQ) = QuadRuleLUT<nQuad>::ab[nQ];
	  h_wt(nQ) = QuadRuleLUT<nQuad>::wt[nQ];
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
