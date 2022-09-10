#include <complex>
#include <cmath>

#include <cerf.h>

#include "EvalFunctionals.H"
#include "CSEKernel.H"

namespace KFVM {

  namespace Stencil {

    namespace SE {

      // Wrapper to pass std::complex<double> to cerf
      namespace {
	std::complex<double> sc_cerf(std::complex<double> z)
	{
	  double _Complex zc99;
	  memcpy(&zc99,&z,sizeof(double _Complex));
	  double _Complex erfc99 = cerf(zc99);
	  std::complex<double> ret;
	  memcpy(&ret,&erfc99,sizeof(double _Complex));
	  return ret;
	}
      }

      // One dimensional kernel functions
      std::complex<double> K(EvalFunctional::Point ef,std::complex<double> eps,const double dx)
      {
	(void) ef;
	return std::exp(-eps*eps*dx*dx);
      }
      
      std::complex<double> K(EvalFunctional::Average ef,std::complex<double> eps,const double dx)
      {
	(void) ef;
	auto pf = std::sqrt(M_PI)/(2.0*eps);
	return pf*(sc_cerf(eps*(dx + 0.5)) - sc_cerf(eps*(dx - 0.5)));
      }
      
      std::complex<double> K(EvalFunctional::Deriv ef,std::complex<double> eps,const double dx)
      {
	(void) ef;
	auto epep = eps*eps;
	return -2.0*epep*dx*std::exp(-epep*dx*dx);
      }
      
      std::complex<double> K(EvalFunctional::SecDeriv ef,std::complex<double> eps,const double dx)
      {
	(void) ef;
	auto epep = eps*eps;
	return 2.0*epep*(2.0*epep*dx*dx - 1.0)*std::exp(-epep*dx*dx);
      }
            
    } // end namespace SE
    
  } // end namespace Stencil
  
} // end namespace KFVM

