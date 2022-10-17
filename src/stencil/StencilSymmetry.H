#ifndef STENCILSYMMETRY_H_
#define STENCILSYMMETRY_H_

#include <cstdio>
#include <utility>
#include <array>
#include <vector>

#include "../Types.H"

namespace KFVM {

  namespace Stencil {

    // The needed symmetries are just combinations of reflections and rotations
    enum class SymType {RefX,RefY,RefZ,RefXY,RefXZ,RefYZ,RefXYZ,RotXY,RotXZ,RotYZ,None};
    // This array is needed because iterating on enums is impossible,
    // but an enum is nice for this purpose otherwise
    constexpr std::array<SymType,11> SymTypes = {SymType::RefX,SymType::RefY,SymType::RefZ,
						 SymType::RefXY,SymType::RefXZ,SymType::RefYZ,
						 SymType::RefXYZ,
						 SymType::RotXY,SymType::RotXZ,SymType::RotYZ,
						 SymType::None};

    // Convert symmetry type to string for printing
    const char* symName(SymType);

    // Aggregate test
    bool testSym(SymType st,double x1,double x2,double y1,double y2);
    bool testSym(SymType st,double x1,double x2,double y1,double y2,double z1,double z2);

    // Pair of reference point, and what symmetry operation gets there
    typedef std::pair<idx_t,SymType> SymMap;

    struct StencilSymmetry {
      // Index maps for each point according to its symmetry relation
      idx_t nPts,N;
      std::vector<SymType> stenSym;
      std::vector<SymMap> symMap;
      std::vector<std::vector<idx_t>> idxMap;
      
      StencilSymmetry() = delete;

      StencilSymmetry(bool isSelf,
		      KFVM_D_DECL(const std::vector<double>&,
				  const std::vector<double>&,
				  const std::vector<double>&),
		      KFVM_D_DECL(const std::vector<double>&,
				  const std::vector<double>&,
				  const std::vector<double>&),
		      KFVM_D_DECL(const std::vector<double>&,
				  const std::vector<double>&,
				  const std::vector<double>&),
		      KFVM_D_DECL(const std::vector<double>&,
				  const std::vector<double>&,
				  const std::vector<double>&));
      
      // Map one set of weights onto another
      template<class DstView,class RefView>
      void mapWeights(DstView& dWts,const RefView& rWts)
      {
	for (int nP=0; nP<nPts; nP++) {
	  if (symMap[nP].first >= 0) {
	    // Fill destination from reference
	    std::printf("    (MW) Filling point %d from %d via %s\n",
			nP,symMap[nP].first,symName(symMap[nP].second));
	    for (int j=0; j<N; j++) {
	      dWts(nP,j) = rWts(symMap[nP].first,idxMap[nP][j]);
	    }
	  } else {
	    std::printf("Symmetry missing in StencilSymmetry::mapWeights\n");
	    std::printf("  (%d,%s) | (%d,%s) | (%d,%s)\n",
			symMap[0].first,symName(symMap[0].second),
			symMap[1].first,symName(symMap[1].second),
			symMap[2].first,symName(symMap[2].second));
	  }
	}
      }
    };
    
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
