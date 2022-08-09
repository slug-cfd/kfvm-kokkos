#ifndef HERMITEPOLYNOMIALS_H_
#define HERMITEPOLYNOMIALS_H_

#include <cstdio>
#include <iostream>
#include <vector>

namespace KFVM {

  namespace Stencil {

    class HermitePolynomials
    {
      int maxDeg;
      std::vector<std::vector<int> > coeffs;

    public:
      HermitePolynomials(int maxDeg_):
	maxDeg(maxDeg_),
	coeffs(maxDeg + 1)
      {
	// allocate all coefficient arrays
	for (int n=0; n<=maxDeg; n++) {
	  coeffs[n].resize(maxDeg+1,0);
	}

	// Coefficients for constant and linear
	coeffs[0][0] = 1;
	coeffs[1][1] = 2;

	// Set remaining coefficients from recurrence
	for (int n=2; n<=maxDeg; n++) {
	  coeffs[n][0] = -coeffs[n-1][1];
	  for (int k=1; k<=n; k++) {
	    coeffs[n][k] = 2*coeffs[n-1][k-1] - (k + 1)*coeffs[n-1][k+1];
	  }
	}
      }

      // Print coefficients for debugging
      void print()
      {
	std::printf("Hermite coefficients to degree: %d\n",maxDeg);
	for (int nD=0; nD<=maxDeg; nD++) {
	  std::printf(" D = %d:",nD);
	  for (int nC=0; nC<=nD; nC++) {
	    std::cout << " " << coeffs[nD][nC] << " |";
	  }
	  std::printf("\n");
	}
      }

      // evaluate nth Hermite polynomial using Horner's method
      double operator()(double x,int n) const
      {
	double v = 0.0;
	for(int k=n; k>=0; k--) {
	  v = v*x + coeffs[n][k];
	}
	return v;
      }
    };
    
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
