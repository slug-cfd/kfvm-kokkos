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
    template<uint16_t nQuad> struct QuadRuleLUT
    {
      static std::array<Real,nQuad> ab,wt;
    };

    // Quadrature rule with values accessible in non-host space
    template<uint16_t nQuad> struct QuadRule
    {
      Kokkos::View<Real[nQuad]> ab,wt;
      QuadRule(): ab("QuadRuleAbscissae"),wt("QuadRuleWeights")
      {
	// Create unmanaged views to wrap LUT values above
	Kokkos::View<Real[nQuad]> h_ab(QuadRuleLUT<nQuad>::ab.data());
	Kokkos::View<Real[nQuad]> h_wt(QuadRuleLUT<nQuad>::wt.data());

	// Copy to intermediate views on the device
	// This seems stupid
	auto ab_tmp = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(),h_ab);
	auto wt_tmp = Kokkos::create_mirror_view_and_copy(Kokkos::DefaultExecutionSpace::memory_space(),h_wt);

	// Copy to the actual views
	Kokkos::deep_copy(ab,ab_tmp);
	Kokkos::deep_copy(wt,wt_tmp);
      }
    };

    template<uint16_t nStage> struct SSPCoeff;
    
    // See Ruuth-Spiteri 2002
    // https://doi.org/10.1137/S0036142901389025
    // This is scheme SSP(5,4) given in appendix A
    template<> struct SSPCoeff<5>
    {
      constexpr static std::array<std::array<Real,5>,5> alpha =
      { std::array<Real,5> {Real(1.0),0.0,0.0,0.0,0.0},
        std::array<Real,5> {Real(0.44437049406734),Real(0.55562950593266),0.0,0.0,0.0},
        std::array<Real,5> {Real(0.62010185138540),0.0,Real(0.37989814861460),0.0,0.0},
        std::array<Real,5> {Real(0.17807995410773),0.0,0.0,Real(0.82192004589227),0.0},
        std::array<Real,5> {
          Real(0.00683325884039),0.0,Real(0.51723167208978),
          Real(0.12759831133288),Real(0.34833675773694)}
      };
      constexpr static std::array<std::array<Real,5>,5> beta =
      { std::array<Real,5> {Real(0.39175222700392),0.0,0.0,0.0,0.0},
        std::array<Real,5> {0.0,Real(0.36841059262959),0.0,0.0,0.0},
        std::array<Real,5> {0.0,0.0,Real(0.25189177424738),0.0,0.0},
        std::array<Real,5> {0.0,0.0,0.0,Real(0.54497475021237),0.0},
        std::array<Real,5> {0.0,0.0,0.0,Real(0.08460416338212),Real(0.22600748319395)}
      };
    };
    
  } // End Numeric namespace
  
} // End KFVM namespace

#endif
