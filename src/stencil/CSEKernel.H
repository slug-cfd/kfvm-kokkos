#ifndef CSEKERNEL_H_
#define CSEKERNEL_H_

#include <complex>

namespace KFVM {

  namespace Stencil {

    namespace SE {

      // Declare the various evaluation functionals
      struct EvalFunctional {
	struct Point {};
	struct Average {};
	struct Deriv {};
	struct SecDeriv {};
      };

      // Forward declare one dimensional kernel functions
      std::complex<double> K(EvalFunctional::Point,std::complex<double>,const double);
      std::complex<double> K(EvalFunctional::Average,std::complex<double>,const double);
      std::complex<double> K(EvalFunctional::Deriv,std::complex<double>,const double);
      std::complex<double> K(EvalFunctional::SecDeriv,std::complex<double>,const double);
      
      template<class efX,class efY>
      std::complex<double> K(std::complex<double> eps,const double dx,const double dy)
      {
	return K(efX(),eps,dx)*K(efY(),eps,dy);
      }
      
      template<class efX,class efY,class efZ>
      std::complex<double> K(std::complex<double> eps,const double dx,const double dy,const double dz)
      {
	return K(efX(),eps,dx)*K(efY(),eps,dy)*K(efZ(),eps,dz);
      }
      
    } // end namespace SE
    
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
