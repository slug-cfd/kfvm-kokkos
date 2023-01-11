#ifndef NUMERIC_H_
#define NUMERIC_H_

// File: Numeric.H
// Purpose: Define reference data for various simple numerical methods
// Notes: The quadrature rules are all defined on (-0.5,0.5) with
//        a slightly hacky "/2.0" at the end of each entry

#include <cstdint>
#include <array>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "RKTypes.H"
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

    template<RK_TYPE RKType> struct RKCoeff{};

    // SSP method, 4 stages, order 3, embedded order 2
    template<>
    struct RKCoeff<RK_TYPE::SSP4_3_2>
    {
      constexpr static int nStages = 4;
      constexpr static std::array<Real,4> gam1 = {0.0,1.0,1.0/3.0,1.0};
      constexpr static std::array<Real,4> gam2 = {0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,4> gam3 = {0.0,0.0,2.0/3.0,0.0};
      constexpr static std::array<Real,4> delta = {0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,4> beta = {1.0/2.0,1.0/2.0,1.0/6.0,1.0/2.0};
      constexpr static std::array<Real,4> bhat = {1.0/4.0,1.0/4.0,1.0/4.0,1.0/4.0};
      constexpr static Real bhatfsal = 0.0;
      constexpr static std::array<Real,5> c = {0.0,1.0/2.0,1.0,1.0/2.0};
      constexpr static Real ep1 = 0.8/3.0;
      constexpr static Real ep2 = -0.31/3.0;
    };

    template<>
    struct RKCoeff<RK_TYPE::ThreeStarP>
    {
      constexpr static int nStages = 5;
      constexpr static std::array<Real,5> gam1 = {0.0,
						  2.587771979725733308135192812685323706e-01,
						  -1.324380360140723382965420909764953437e-01,
						  5.056033948190826045833606441415585735e-02,
						  5.670532000739313812633197158607642990e-01};
      constexpr static std::array<Real,5> gam2 = {0.0,
						  5.528354909301389892439698870483746541e-01,
						  6.731871608203061824849561782794643600e-01,
						  2.803103963297672407841316576323901761e-01,
						  5.521525447020610386070346724931300367e-01};
      constexpr static std::array<Real,5> gam3 = {0.0,
						  0.0,
						  0.0,
						  2.752563273304676380891217287572780582e-01,
						  -8.950526174674033822276061734289327568e-01};
      constexpr static std::array<Real,5> delta = {1.0,
						   3.407655879334525365094815965895763636e-01,
						   3.414382655003386206551709871126405331e-01,
						   7.229275366787987419692007421895451953e-01,
						   0.0};
      constexpr static std::array<Real,5> beta = {2.300298624518076223899418286314123354e-01,
						  3.021434166948288809034402119555380003e-01,
						  8.025606185416310937583009085873554681e-01,
						  4.362158943603440930655148245148766471e-01,
						  1.129272530455059129782111662594436580e-01};
      constexpr static std::array<Real,5> bhat = {9.484166705035703392326247283838082847e-02,
						  1.726371339430353766966762629176676070e-01,
						  3.998243189084371024483169698618455770e-01,
						  1.718016807580178450618829007973835152e-01,
						  5.881914422155740300718268359027168467e-02};
      constexpr static Real bhatfsal = 1.020760551185952388626787099944507877e-01;
      constexpr static std::array<Real,5> c = {0.0,
					       2.300298624518076223899418286314123354e-01,
					       4.050046072094990912268498160116125481e-01,
					       8.947822893693433545220710894560512805e-01,
					       7.235136928826589010272834603680114769e-01};
      constexpr static Real ep1 = 0.7/3.0;
      constexpr static Real ep2 = -0.23/3.0;
    };

    template<>
    struct RKCoeff<RK_TYPE::FourStarP>
    {
      constexpr static int nStages = 9;
      constexpr static std::array<Real,9> gam1 = {0.0,
						  -4.655641447335068552684422206224169103e+00,
						  -7.720265099645871829248487209517314217e-01,
						  -4.024436690519806086742256154738379161e+00,
						  -2.129676284018530966221583708648634733e-02,
						  -2.435022509790109546199372365866450709e+00,
						  1.985627297131987000579523283542615256e-02,
						  -2.810791146791038566946663374735713961e-01,
						  1.689434168754859644351230590422137972e-01};
      constexpr static std::array<Real,9> gam2 = {0.0,
						  2.499262792574495009336242992898153462e+00,
						  5.866820377718875577451517985847920081e-01,
						  1.205146086523094569925592464380295241e+00,
						  3.474793722186732780030762737753849272e-01,
						  1.321346060965113109321230804210670518e+00,
						  3.119636464694193615946633676950358444e-01,
						  4.351419539684379261368971206040518552e-01,
						  2.359698130028753572503744518147537768e-01};
      constexpr static std::array<Real,9> gam3 = {0.0,
						  0.0,
						  0.0,
						  7.621006678721315291614677352949377871e-01,
						  -1.981182504339400567765766904309673119e-01,
						  -6.228959218699007450469629366684127462e-01,
						  -3.752248380775956442989480369774937099e-01,
						  -3.355438309135169811915662336248989661e-01,
						  -4.560955005031121479972862973705108039e-02};
      constexpr static std::array<Real,9> delta = {1.0,
						   1.262923876648114432874834923838556100e+00,
						   7.574967189685911558308119415539596711e-01,
						   5.163589453140728104667573195005629833e-01,
						   -2.746327421802609557034437892013640319e-02,
						   -4.382673178127944142238606608356542890e-01,
						   1.273587294602656522645691372699677063e+00,
						   -6.294740283927400326554066998751383342e-01,
						   0.0};
      constexpr static std::array<Real,9> beta = {2.836343005184365275160654678626695428e-01,
						  9.736500104654741223716056170419660217e-01,
						  3.382359225242515288768487569778320563e-01,
						  -3.584943611106183357043212309791897386e-01,
						  -4.113944068471528211627210454497620358e-03,
						  1.427968894048586363415504654313371031e+00,
						  1.808470948394314017665968411915568633e-02,
						  1.605770645946802213926893453819236685e-01,
						  2.952227015964591648775833803635147962e-01};
      constexpr static std::array<Real,9> bhat = {2.483675912451591196775756814283216443e-02,
						  1.866327774562103796990092260942180726e-01,
						  5.671080795936984495604436622517631183e-02,
						  -3.447695439149287702616943808570747099e-03,
						  3.602245056516636472203469198006404016e-03,
						  4.545570622145088936800484247980581766e-01,
						  -2.434665289427612407531544765622888855e-04,
						  6.642755361103549971517945063138312147e-02,
						  1.613697079523505006226025497715177578e-01};
      constexpr static Real bhatfsal = 4.955424859358438183052504342394102722e-02;
      constexpr static std::array<Real,9> c = {0.0,
					       2.836343005184365275160654678626695428e-01,
					       5.484076570002894365286665352032296535e-01,
					       3.687228761669438493478872632332010073e-01,
					       -6.806126440140844191258463830024463902e-01,
					       3.518526124230705801739919476290327750e-01,
					       1.665941994879593315477304663913129942e+00,
					       9.715279295934715835299192116436237065e-01,
					       9.051569840159589594903399929316959062e-01};
      constexpr static Real ep1 = 0.38/4.0;
      constexpr static Real ep2 = -0.18/4.0;
    };    

    template<>
    struct RKCoeff<RK_TYPE::KETCH104>
    {
      constexpr static int nStages = 10;
      constexpr static std::array<Real,10> gam1 = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,10> gam2 = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,10> gam3 = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,10> delta = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      constexpr static std::array<Real,10> beta = {1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,
                                                   1.0/6.0,1.0/6.0,1.0/6.0,1.0/6.0,1.0/10.0};
      constexpr static std::array<Real,10> bhat = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
      constexpr static Real bhatfsal = 0.0;
      constexpr static std::array<Real,10> c = {0.0,
                                                1.0/6.0,
                                                1.0/3.0,
                                                1.0/2.0,
                                                2.0/3.0,
                                                1.0/3.0,
                                                1.0/2.0,
                                                2.0/3.0,
                                                5.0/6.0,
                                                1.0};
      constexpr static Real ep1 = 0.0;
      constexpr static Real ep2 = 0.0;
    };

  } // End Numeric namespace
  
} // End KFVM namespace

#endif
