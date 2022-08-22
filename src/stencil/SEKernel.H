#ifndef SEKERNEL_H_
#define SEKERNEL_H_

#include <cmath>

#include <Definitions.H>

#include "../Types.H"
#include "HermitePolynomials.H"

namespace KFVM {

  namespace Stencil {

    namespace SE {

      struct PhiFunctional
      {
	struct Point {};
	struct Average {};
	struct Deriv {};
	struct SecDeriv {};
      };

      // HS eigenvalues for (univariate) squared exponential
      inline double eig(int n,double eps,double alpha,double delsq)
      {
	double ade = alpha*alpha + delsq + eps*eps;
	return sqrt(alpha*alpha/ade)*pow(eps*eps/ade,n);
      }

      // weighted integral of Hermite polynomial
      inline double hermInt(int n,const HermitePolynomials& herm,
			    double dabsq,double yup,double ylo)
      {
	if (n == 0) {
	  double dab = sqrt(dabsq);
	  return sqrt(M_PI)*(erf(dab*yup) - erf(dab*ylo))/(2.0*dab);
	}
	else if (n == 1) {
	  return (exp(-dabsq*ylo*ylo) - exp(-dabsq*yup*yup))/dabsq;
	}
	double bdy = herm(ylo,n - 1)*exp(-dabsq*ylo*ylo) - herm(yup,n - 1)*exp(-dabsq*yup*yup);
	double hInt = hermInt(n - 2,herm,dabsq,yup,ylo);
	return (bdy + 2.0*(1.0 - dabsq)*(n - 1)*hInt)/dabsq;
      }

      // Pointwise SE eigenfunctions
      inline double phi(PhiFunctional::Point ft,
			double x,const HermitePolynomials& herm,int n,
			double alpha,double beta,double delsq)
      {
	(void) ft;
	// recall that gamma(n + 1) == n!
	double gam = sqrt(beta/(pow(2.0,n) * tgamma(n + 1)));
	double ab = alpha*beta;
	double scl = exp(-delsq*x*x);
	return gam*scl*herm(ab*x,n);
      }

      // integrated SE eigenfunctions
      inline double phi(PhiFunctional::Average ft,
			double x,const HermitePolynomials& herm,int n,
			double alpha,double beta,double delsq)
      {
	(void) ft;
	// recall that gamma(n + 1) == n!
	double gam = sqrt(beta/(pow(2.0,n) * tgamma(n + 1)));
	double ab = alpha*beta;
	double dabsq = delsq/(ab*ab);
	double yup = ab*(x + 0.5);
	double ylo = ab*(x - 0.5);
	return gam*hermInt(n,herm,dabsq,yup,ylo)/ab;
      }

      // Derivative SE eigenfunctions
      inline double phi(PhiFunctional::Deriv ft,
			double x,const HermitePolynomials& herm,int n,
			double alpha,double beta,double delsq)
      {
	(void) ft;
	// recall that gamma(n + 1) == n!
	double gam = sqrt(beta/(pow(2.0,n) * tgamma(n + 1)));
	double ab = alpha*beta;
	double scl = exp(-delsq*x*x);
	return 2.0*gam*scl*(n*ab*herm(ab*x,n - 1) - delsq*x*herm(ab*x,n));
      }

      // Second Derivative SE eigenfunctions
      inline double phi(PhiFunctional::SecDeriv ft,
			double x,const HermitePolynomials& herm,int n,
			double alpha,double beta,double delsq)
      {
	(void) ft;
	// recall that gamma(n + 1) == n!
	double gam = sqrt(beta/(pow(2.0,n) * tgamma(n + 1)));
	double ab = alpha*beta;
	double scl = exp(-delsq*x*x);
	return 2.0*gam*scl*(2.0*n*(n - 1)*ab*ab*herm(ab*x,n - 2) -
			    2.0*(n + 1)*ab*delsq*x*herm(ab*x,n - 1) +
			    delsq*(2.0*delsq*x*x - 1.0)*herm(ab*x,n));
      }
      
    } // end namespace SE
    
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
