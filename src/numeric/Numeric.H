#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <array>

namespace KFVM {

  namespace Numeric {
    
    template<std::size_t nQuad> struct QuadRule;

    // Quadrature rules with 1-9 points
    // All are defined on the interval (-1,1)
    template<> struct QuadRule<1>
    {
      constexpr static std::array<Real,1> ab =
      {Real(0.0)};
      constexpr static std::array<Real,1> wt =
      {Real(2.0)};
    };
  
    template<> struct QuadRule<2>
    {
      constexpr static std::array<Real,2> ab =
      {Real(-0.577350269189625764509148),
       Real(0.577350269189625764509148)};
      constexpr static std::array<Real,2> wt =
      {Real(1.0),
       Real(1.0)};
    };
  
    template<> struct QuadRule<3>
    {
      constexpr static std::array<Real,3> ab =
      {Real(0.0),
       Real(-0.7745966692414833770358530),
       Real(0.7745966692414833770358530)};
      constexpr static std::array<Real,3> wt =
      {Real(0.8888888888888888888888888),
       Real(0.5555555555555555555555555),
       Real(0.5555555555555555555555555)};
    };
  
    template<> struct QuadRule<4>
    {
      constexpr static std::array<Real,4> ab =
      {Real(-0.339981043584856264802665),
       Real(0.339981043584856264802665),
       Real(-0.861136311594052575223946),
       Real(0.861136311594052575223946)};
      constexpr static std::array<Real,4> wt =
      {Real(0.6521451548625461426269360),
       Real(0.6521451548625461426269360),
       Real(0.3478548451374538573730639),
       Real(0.3478548451374538573730639)};
    };
  
    template<> struct QuadRule<5>
    {
      constexpr static std::array<Real,5> ab =
      {Real(0.0),
       Real(-0.5384693101056830910363144),
       Real(0.5384693101056830910363144),
       Real(-0.9061798459386639927976268),
       Real(0.9061798459386639927976268)};
      constexpr static std::array<Real,5> wt =
      {Real(0.5688888888888888888888888),
       Real(0.4786286704993664680412915),
       Real(0.4786286704993664680412915),
       Real(0.2369268850561890875142640),
       Real(0.2369268850561890875142640)};
    };
  
    template<> struct QuadRule<6>
    {
      constexpr static std::array<Real,6> ab =
      {Real(-0.2386191860831969086305017),
       Real(0.2386191860831969086305017),
       Real(-0.6612093864662645136613995),
       Real(0.6612093864662645136613995),
       Real(-0.9324695142031520278123015),
       Real(0.9324695142031520278123015)};
      constexpr static std::array<Real,6> wt =
      {Real(0.4679139345726910473898703),
       Real(0.4679139345726910473898703),
       Real(0.3607615730481386075698335),
       Real(0.3607615730481386075698335),
       Real(0.1713244923791703450402961),
       Real(0.1713244923791703450402961)};
    };
  
    template<> struct QuadRule<7>
    {
      constexpr static std::array<Real,7> ab =
      {Real(0.0),
       Real(0.4058451513773971669066064),
       Real(-0.4058451513773971669066064),
       Real(-0.7415311855993944398638647),
       Real(0.7415311855993944398638647),
       Real(-0.9491079123427585245261896),
       Real(0.9491079123427585245261896)};
      constexpr static std::array<Real,7> wt =
      {Real(0.4179591836734693877551020),
       Real(0.3818300505051189449503697),
       Real(0.3818300505051189449503697),
       Real(0.2797053914892766679014677),
       Real(0.2797053914892766679014677),
       Real(0.1294849661688696932706114),
       Real(0.1294849661688696932706114)};
    };
  
    template<> struct QuadRule<8>
    {
      constexpr static std::array<Real,8> ab =
      {Real(-0.183434642495649804939476),
       Real(0.183434642495649804939476),
       Real(-0.525532409916328985817739),
       Real(0.525532409916328985817739),
       Real(-0.796666477413626739591553),
       Real(0.796666477413626739591553),
       Real(-0.960289856497536231683560),
       Real(0.960289856497536231683560)};
      constexpr static std::array<Real,8> wt =
      {Real(0.3626837833783619829651504),
       Real(0.3626837833783619829651504),
       Real(0.3137066458778872873379622),
       Real(0.3137066458778872873379622),
       Real(0.2223810344533744705443559),
       Real(0.2223810344533744705443559),
       Real(0.1012285362903762591525313),
       Real(0.1012285362903762591525313)};
    };
  
    template<> struct QuadRule<9>
    {
      constexpr static std::array<Real,9> ab =
      {Real(0.0),
       Real(-0.8360311073266357942994297),
       Real(0.8360311073266357942994297),
       Real(-0.9681602395076260898355762),
       Real(0.9681602395076260898355762),
       Real(-0.3242534234038089290385380),
       Real(0.3242534234038089290385380),
       Real(-0.6133714327005903973087020),
       Real(0.6133714327005903973087020)};
      constexpr static std::array<Real,9> wt =
      {Real(0.3302393550012597631645250),
       Real(0.1806481606948574040584720),
       Real(0.1806481606948574040584720),
       Real(0.0812743883615744119718921),
       Real(0.0812743883615744119718921),
       Real(0.3123470770400028400686304),
       Real(0.3123470770400028400686304),
       Real(0.2606106964029354623187428),
       Real(0.2606106964029354623187428)};
    };

    template<std::size_t S> struct SSPCoeff;
    
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
