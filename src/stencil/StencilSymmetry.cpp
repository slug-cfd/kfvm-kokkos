#include "../Types.H"
#include "StencilSymmetry.H"


namespace KFVM {

  namespace Stencil {

    // Conversion from symmetry type to name for printing
    const char* symName(SymType st)
    {
      switch (st) {
      case SymType::RefX:
	return "RefX";
      case SymType::RefY:
	return "RefY";
      case SymType::RefZ:
	return "RefZ";
      case SymType::RefXY:
	return "RefXY";
      case SymType::RefXZ:
	return "RefXZ";
      case SymType::RefYZ:
	return "RefYZ";
      case SymType::RefXYZ:
	return "RefXYZ";
      case SymType::RotXY:
	return "RotXY";
      case SymType::RotXZ:
	return "RotXZ";
      case SymType::RotYZ:
	return "RotYZ";
      default:
	return "None";
      }
    }

    // Functions to test for symmetry
    template<SymType st> bool testSymP(double,double,double,double);
    template<SymType st> bool testSymP(double,double,double,double,double,double);

    // Single reflections
    template<>
    bool testSymP<SymType::RefX>(double x1,double x2,double y1,double y2)
    {
      return (x1 == -x2) && (y1 == y2);
    }

    template<>    
    bool testSymP<SymType::RefX>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == -x2) && (y1 == y2) && (z1 == z2);
    }

    template<>    
    bool testSymP<SymType::RefY>(double x1,double x2,double y1,double y2)
    {
      return (x1 == x2) && (y1 == -y2);
    }

    template<>    
    bool testSymP<SymType::RefY>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == x2) && (y1 == -y2) && (z1 == z2);
    }

    template<>    
    bool testSymP<SymType::RefZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == x2) && (y1 == y2) && (z1 == -z2);
    }

    // Double reflections
    template<>
    bool testSymP<SymType::RefXY>(double x1,double x2,double y1,double y2)
    {
      return (x1 == -x2) && (y1 == -y2);
    }

    template<>    
    bool testSymP<SymType::RefXY>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == -x2) && (y1 == -y2) && (z1 == z2);
    }

    template<>    
    bool testSymP<SymType::RefXZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == -x2) && (y1 == y2) && (z1 == -z2);
    }

    template<>    
    bool testSymP<SymType::RefYZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == x2) && (y1 == -y2) && (z1 == -z2);
    }

    // Triple reflection
    template<>
    bool testSymP<SymType::RefXYZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == -x2) && (y1 == -y2) && (z1 == -z2);
    }

    // Planar rotations
    template<>
    bool testSymP<SymType::RotXY>(double x1,double x2,double y1,double y2)
    {
      return (x1 == -y2) && (y1 == x2);
    }

    template<>    
    bool testSymP<SymType::RotXY>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == -y2) && (y1 == x2) && (z1 == z2);
    }

    template<>    
    bool testSymP<SymType::RotXZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == z2) && (y1 == y2) && (z1 == x2);
    }

    template<>    
    bool testSymP<SymType::RotYZ>(double x1,double x2,double y1,double y2,double z1,double z2)
    {
      return (x1 == x2) && (y1 == z2) && (z1 == y1);
    }

    // Aggregate test
    bool testSym(SymType st,double x1,double x2,double y1,double y2)
    {
      switch (st) {
      case SymType::RefX:
	return testSymP<SymType::RefX>(x1,x2,y1,y2);
      case SymType::RefY:
	return testSymP<SymType::RefY>(x1,x2,y1,y2);
      case SymType::RefXY:
	return testSymP<SymType::RefXY>(x1,x2,y1,y2);
      case SymType::RotXY:
	return testSymP<SymType::RotXY>(x1,x2,y1,y2);
      default:
	return false;
      }
    }
    
    bool testSym(SymType st,double x1,double x2,double y1,double y2,double z1,double z2)
    {
      switch (st) {
      case SymType::RefX:
	return testSymP<SymType::RefX>(x1,x2,y1,y2,z1,z2);
      case SymType::RefY:
	return testSymP<SymType::RefY>(x1,x2,y1,y2,z1,z2);
      case SymType::RefZ:
	return testSymP<SymType::RefZ>(x1,x2,y1,y2,z1,z2);
      case SymType::RefXY:
	return testSymP<SymType::RefXY>(x1,x2,y1,y2,z1,z2);
      case SymType::RefXZ:
	return testSymP<SymType::RefXZ>(x1,x2,y1,y2,z1,z2);
      case SymType::RefYZ:
	return testSymP<SymType::RefYZ>(x1,x2,y1,y2,z1,z2);
      case SymType::RefXYZ:
	return testSymP<SymType::RefXYZ>(x1,x2,y1,y2,z1,z2);
      case SymType::RotXY:
	return testSymP<SymType::RotXY>(x1,x2,y1,y2,z1,z2);
      case SymType::RotXZ:
	return testSymP<SymType::RotXZ>(x1,x2,y1,y2,z1,z2);
      case SymType::RotYZ:
	return testSymP<SymType::RotYZ>(x1,x2,y1,y2,z1,z2);
      default:
	return false;
      }
    }
    
    // Function to find symmetries present within a set of points
    std::vector<SymMap> symmetryRelations(const std::vector<Real>& x,
					  const std::vector<Real>& y)
    {
      idx_t nPts = x.size();
      
      // Default the relations to no references and no symmetries
      std::vector<SymMap> symRel(nPts,{-1,SymType::None});
      
      // Loop over points starting from the second
      for (idx_t nD=1; nD<nPts; nD++) {
	// and try to map them onto earlier ones
	for (idx_t nR=0; nR<nD; nR++) {
	  // Test each symmetry in 2D
	  if (testSymP<SymType::RefX>(x[nD],x[nR],y[nD],y[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefX;
	    break;
	  } else if (testSymP<SymType::RefY>(x[nD],x[nR],y[nD],y[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefY;
	    break;
	  } else if (testSymP<SymType::RefXY>(x[nD],x[nR],y[nD],y[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXY;
	    break;
	  } else if (testSymP<SymType::RotXY>(x[nD],x[nR],y[nD],y[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXY;
	    break;
	  }
	}
      }

      return symRel;
    }

    // Function to find symmetries present within a set of points
    std::vector<SymMap> symmetryRelations(const std::vector<Real>& x,
					  const std::vector<Real>& y,
					  const std::vector<Real>& z)
    {
      idx_t nPts = x.size();
      
      // Default the relations to no references and no symmetries
      std::vector<SymMap> symRel(nPts,{-1,SymType::None});
      
      // Loop over points starting from the second
      for (idx_t nD=1; nD<nPts; nD++) {
	// and try to map them onto earlier ones
	for (idx_t nR=0; nR<nD; nR++) {
	  // Test each symmetry in 2D
	  if (testSymP<SymType::RefX>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefX;
	    break;
	  } else if (testSymP<SymType::RefY>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefY;
	    break;
	  } else if (testSymP<SymType::RefZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefZ;
	    break;
	  } else if (testSymP<SymType::RefXY>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXY;
	    break;
	  } else if (testSymP<SymType::RefXZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXZ;
	    break;
	  } else if (testSymP<SymType::RefYZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefYZ;
	    break;
	  } else if (testSymP<SymType::RefXYZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXYZ;
	    break;
	  } else if (testSymP<SymType::RotXY>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXY;
	    break;
	  } else if (testSymP<SymType::RotXZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXZ;
	    break;
	  } else if (testSymP<SymType::RotYZ>(x[nD],x[nR],y[nD],y[nR],z[nD],z[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotYZ;
	    break;
	  }
	}
      }

      return symRel;
    }
    
    // Function to find symmetries between two sets of points
    std::vector<SymMap> symmetryRelations(const std::vector<Real>& xd,
					  const std::vector<Real>& yd,
					  const std::vector<Real>& xr,
					  const std::vector<Real>& yr)
    {
      idx_t nPts = xd.size();

      if (nPts == 3) {
	std::printf("Mapping point set:\n");
	std::printf("  (%lf,%lf) | (%lf,%lf) | (%lf,%lf)\n",
		    xr[0],yr[0],xr[1],yr[1],xr[2],yr[2]);
	std::printf("onto:\n");
	std::printf("  (%lf,%lf) | (%lf,%lf) | (%lf,%lf)\n",
		    xd[0],yd[0],xd[1],yd[1],xd[2],yd[2]);
      }
      
      // Default the relations to no references and no symmetries
      std::vector<SymMap> symRel(nPts,{-1,SymType::None});
      
      // Loop over destination points
      for (idx_t nD=0; nD<nPts; nD++) {
	// and try to map them onto reference points
	for (idx_t nR=0; nR<nPts; nR++) {
	  // Test each symmetry in 2D
	  if (testSymP<SymType::RefX>(xd[nD],xr[nR],yd[nD],yr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefX;
	    break;
	  } else if (testSymP<SymType::RefY>(xd[nD],xr[nR],yd[nD],yr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefY;
	    break;
	  } else if (testSymP<SymType::RefXY>(xd[nD],xr[nR],yd[nD],yr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXY;
	    break;
	  } else if (testSymP<SymType::RotXY>(xd[nD],xr[nR],yd[nD],yr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXY;
	    break;
	  }
	}
      }

      if (nPts == 3) {
	std::printf("Found map:\n");
	std::printf("  (%d,%s) | (%d,%s) | (%d,%s)\n",
		    symRel[0].first,symName(symRel[0].second),
		    symRel[1].first,symName(symRel[1].second),
		    symRel[2].first,symName(symRel[2].second));
      }

      return symRel;
    }

    // Function to find symmetries present within a set of points
    std::vector<SymMap> symmetryRelations(const std::vector<Real>& xd,
                                          const std::vector<Real>& yd,
                                          const std::vector<Real>& zd,
					  const std::vector<Real>& xr,
					  const std::vector<Real>& yr,
					  const std::vector<Real>& zr)
    {
      idx_t nPts = xd.size();
      
      // Default the relations to no references and no symmetries
      std::vector<SymMap> symRel(nPts,{-1,SymType::None});
      
      // Loop over destination points
      for (idx_t nD=0; nD<nPts; nD++) {
	// and try to map them onto references
	for (idx_t nR=0; nR<nPts; nR++) {
	  // Test each symmetry in 3D
	  if (testSymP<SymType::RefX>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefX;
	    break;
	  } else if (testSymP<SymType::RefY>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefY;
	    break;
	  } else if (testSymP<SymType::RefZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefZ;
	    break;
	  } else if (testSymP<SymType::RefXY>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXY;
	    break;
	  } else if (testSymP<SymType::RefXZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXZ;
	    break;
	  } else if (testSymP<SymType::RefYZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefYZ;
	    break;
	  } else if (testSymP<SymType::RefXYZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RefXYZ;
	    break;
	  } else if (testSymP<SymType::RotXY>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXY;
	    break;
	  } else if (testSymP<SymType::RotXZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotXZ;
	    break;
	  } else if (testSymP<SymType::RotYZ>(xd[nD],xr[nR],yd[nD],yr[nR],zd[nD],zr[nR])) {
	    symRel[nD].first = nR;
	    symRel[nD].second = SymType::RotYZ;
	    break;
	  }
	}
      }

      return symRel;
    }

  } // end namespace Stencil

} // end namespace KFVM
