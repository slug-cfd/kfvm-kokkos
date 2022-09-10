#ifndef MONONOMIALS_H_
#define MONONOMIALS_H_

#include <array>
#include <complex>

#include "EvalFunctionals.H"

namespace KFVM {

  namespace Stencil {

    namespace Monomials {

      // Declare univariate monomials under each evaluation functional
      // Note: These are purely real, but need to return complex
      std::complex<double> mono(EvalFunctional::Point,int,const double);
      std::complex<double> mono(EvalFunctional::Average,int,const double);
      std::complex<double> mono(EvalFunctional::Deriv,int,const double);
      std::complex<double> mono(EvalFunctional::SecDeriv,int,const double);

      // Define multivariate monomials through 1d calls
      template<class efX,class efY>
      std::complex<double> mono(const std::array<int,2>& p,
				const double x,const double y)
      {
	return mono(efX(),p[0],x)*mono(efY(),p[1],y);
      }
      
      template<class efX,class efY,class efZ>
      std::complex<double> mono(const std::array<int,3>& p,
				const double x,const double y,const double z)
      {
	return mono(efX(),p[0],x)*mono(efY(),p[1],y)*mono(efZ(),p[2],z);
      }

    } // end namespace Monomials
      
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
