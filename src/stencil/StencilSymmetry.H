#ifndef STENCILSYMMETRY_H_
#define STENCILSYMMETRY_H_

#include <cstdio>
#include <utility>
#include <vector>

#include "../Types.H"

namespace KFVM {

  namespace Stencil {

    // The needed symmetries are just combinations of reflections
    enum class SymType {RefX,RefY,RefZ,RefXY,RefXZ,RefYZ,RefXYZ,RotXY,RotXZ,RotYZ,None};

    // Convert symmetry type to string for printing
    const char* symName(SymType);

    // Aggregate test
    bool testSym(SymType st,double x1,double x2,double y1,double y2);
    bool testSym(SymType st,double x1,double x2,double y1,double y2,double z1,double z2);

    // Pair of reference point, and what symmetry operation gets there
    typedef std::pair<idx_t,SymType> SymMap;

    // Functions to find symmetry relations within a point set
    std::vector<SymMap> symmetryRelations(const std::vector<Real>&,
					  const std::vector<Real>&);
    std::vector<SymMap> symmetryRelations(const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&);
    
    // Functions to find symmetry relations between two point sets
    std::vector<SymMap> symmetryRelations(const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&);
    std::vector<SymMap> symmetryRelations(const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&,
					  const std::vector<Real>&);

    struct StencilSymmetry {
      // Index maps for each point according to its symmetry relation
      idx_t nPts,N;
      std::vector<SymMap> symMap;
      std::vector<std::vector<idx_t>> idxMap;
      
      StencilSymmetry() = delete;

      // Symmetry within a 2d stencil
      StencilSymmetry(const std::vector<SymMap>& symMap_,
		      const std::vector<double>& lOff,
		      const std::vector<double>& tOff):
	nPts(symMap_.size()),
	N(lOff.size()),
	symMap(symMap_),
	idxMap(nPts,std::vector<idx_t>(N,-1))
      {
	// Find mapping for each point
	for (idx_t nP=0; nP<nPts; nP++) {
	  // Skip if this point isn't given by another
	  if (symMap[nP].first < 0) { continue; }

	  // Otherwise
	  bool stenSym = true;
	  for (idx_t nD=0; nD<N; nD++) {
	    bool matched = false;
	    for (idx_t nR=0; nR<N; nR++) {
	      if (testSym(symMap[nP].second,
			  lOff[nD],lOff[nR],
			  tOff[nD],tOff[nR])) {
		idxMap[nP][nD] = nR;
		matched = true;
		break;
	      }
	    }
	    stenSym = stenSym && matched;
	  }
	  
	  // If the stencil does not have the same symmetry
	  // as the point, override the point
	  if (!stenSym) {
	    symMap[nP].first = -1;
	    symMap[nP].second = SymType::None;
	  }
	}
      }

      // Symmetry within a 3d stencil
      StencilSymmetry(const std::vector<SymMap>& symMap_,
		      const std::vector<double>& lOff,
		      const std::vector<double>& tOff,
		      const std::vector<double>& ttOff):
	nPts(symMap_.size()),
	N(lOff.size()),
	symMap(symMap_),
	idxMap(nPts,std::vector<idx_t>(N,-1))
      {
	// Find mapping for each point
	for (idx_t nP=0; nP<nPts; nP++) {
	  // Skip if this point isn't given by another
	  if (symMap[nP].first < 0) { continue; }

	  // Otherwise
	  bool stenSym = true;
	  for (idx_t nD=0; nD<N; nD++) {
	    bool matched = false;
	    for (idx_t nR=0; nR<N; nR++) {
	      if (testSym(symMap[nP].second,
			  lOff[nD],lOff[nR],
			  tOff[nD],tOff[nR],
			  ttOff[nD],ttOff[nR])) {
		idxMap[nP][nD] = nR;
		matched = true;
		break;
	      }
	    }
	    stenSym = stenSym && matched;
	  }
	  
	  // If the stencil does not have the same symmetry
	  // as the point, override the point
	  if (!stenSym) {
	    symMap[nP].first = -1;
	    symMap[nP].second = SymType::None;
	  }
	}
      }

      // Symmetry between two 2d stencils
      StencilSymmetry(const std::vector<SymMap>& symMap_,
		      const std::vector<double>& lOffD,
		      const std::vector<double>& tOffD,
		      const std::vector<double>& lOffR,
		      const std::vector<double>& tOffR):
	nPts(symMap_.size()),
	N(lOffD.size()),
	symMap(symMap_),
	idxMap(nPts,std::vector<idx_t>(N,-1))
      {
	// Find mapping for each point
	for (idx_t nP=0; nP<nPts; nP++) {
	  // Skip if this point isn't given by another
	  if (symMap[nP].first < 0) { continue; }

	  // Otherwise
	  bool stenSym = true;
	  for (idx_t nD=0; nD<N; nD++) {
	    bool matched = false;
	    for (idx_t nR=0; nR<N; nR++) {
	      if (testSym(symMap[nP].second,
			  lOffD[nD],lOffR[nR],
			  tOffD[nD],tOffR[nR])) {
		idxMap[nP][nD] = nR;
		matched = true;
		break;
	      }
	    }
	    stenSym = stenSym && matched;
	  }
	  
	  // If the stencil does not have the same symmetry
	  // as the point, override the point
	  if (!stenSym) {
	    symMap[nP].first = -1;
	    symMap[nP].second = SymType::None;
	  }
	}
      }

      // Symmetry between two 3d stencils
      StencilSymmetry(const std::vector<SymMap>& symMap_,
		      const std::vector<double>& lOffD,
		      const std::vector<double>& tOffD,
		      const std::vector<double>& ttOffD,
		      const std::vector<double>& lOffR,
		      const std::vector<double>& tOffR,
		      const std::vector<double>& ttOffR):
	nPts(symMap_.size()),
	N(lOffD.size()),
	symMap(symMap_),
	idxMap(nPts,std::vector<idx_t>(N,-1))
      {
	// Find mapping for each point
	for (idx_t nP=0; nP<nPts; nP++) {
	  // Skip if this point isn't given by another
	  if (symMap[nP].first < 0) { continue; }

	  // Otherwise
	  bool stenSym = true;
	  for (idx_t nD=0; nD<N; nD++) {
	    bool matched = false;
	    for (idx_t nR=0; nR<N; nR++) {
	      if (testSym(symMap[nP].second,
			  lOffD[nD],lOffR[nR],
			  tOffD[nD],tOffR[nR],
			  ttOffD[nD],ttOffR[nR])) {
		idxMap[nP][nD] = nR;
		matched = true;
		break;
	      }
	    }
	    stenSym = stenSym && matched;
	  }
	  
	  // If the stencil does not have the same symmetry
	  // as the point, override the point
	  if (!stenSym) {
	    symMap[nP].first = -1;
	    symMap[nP].second = SymType::None;
	  }
	}
      }

      // Map one set of weights onto another
      template<class DstView,class RefView>
      void mapWeights(DstView& dWts,const RefView& rWts)
      {
	for (int nP=0; nP<nPts; nP++) {
	  if (symMap[nP].first >= 0) {
	    // Fill destination from reference
	    std::printf("    Filling point %d from %d via %s\n",
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

      // Print out symmetries
      void print()
      {
	for (int nP=0; nP<nPts; nP++) {
	  std::printf("Point %d references point %d via symmetry type %s\n",
		      nP,symMap[nP].first,symName(symMap[nP].second));
	}
      }

      // Print out symmetries
      void print(const std::vector<double>& dxs,
		 const std::vector<double>& dys,
		 const std::vector<double>& rxs,
		 const std::vector<double>& rys)
      {
	for (int nP=0; nP<nPts; nP++) {
	  std::printf("    Point %d references point %d via symmetry type %s\n",
		      nP,symMap[nP].first,symName(symMap[nP].second));
	  for (int j=0; j<N; j++) {
	    std::printf("      (%1.0f,%1.0f) <- (%1.0f,%1.0f)\n",
			dxs[j],dys[j],rxs[idxMap[nP][j]],rys[idxMap[nP][j]]);
	  }
	}
      }
    };
    
  } // end namespace Stencil
  
} // end namespace KFVM

#endif
