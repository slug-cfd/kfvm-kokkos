#include <complex>
#include <cmath>

#include "EvalFunctionals.H"

namespace KFVM {

  namespace Stencil {

    namespace Monomials {
      
      std::complex<double> mono(EvalFunctional::Point ef,int p,const double x)
      {
	(void) ef;
	return std::complex<double>(std::pow(x,p),0.0);
      }
      
      std::complex<double> mono(EvalFunctional::Average ef,int p,const double x)
      {
	(void) ef;
	double ub = std::pow(x + 0.5,p + 1)/(p + 1);
	double lb = std::pow(x - 0.5,p + 1)/(p + 1);
	return std::complex<double>(ub - lb,0.0);
      }
      
      std::complex<double> mono(EvalFunctional::Deriv ef,int p,const double x)
      {
	(void) ef;
	if (p > 0) {
	  return std::complex<double>(p*std::pow(x,p - 1),0.0);
	} else {
	  return std::complex<double>(0.0,0.0);
	}
      }
      
      std::complex<double> mono(EvalFunctional::SecDeriv ef,int p,const double x)
      {
	(void) ef;
	if (p > 1) {
	  return std::complex<double>(p*(p - 1)*std::pow(x,p - 2),0.0);
	} else {
	  return std::complex<double>(0.0,0.0);
	}
      }
      
    } // end namespace Monomials
      
  } // end namespace Stencil
  
} // end namespace KFVM
