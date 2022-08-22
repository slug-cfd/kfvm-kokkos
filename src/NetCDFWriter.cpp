#include <Kokkos_CopyViews.hpp>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>

#include <netcdf>

#include <Definitions.H>

#include "Dimension.H"
#include "Geometry.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "numeric/Numeric.H"
#include "NetCDFWriter.H"

namespace KFVM {

  NetCDFWriter::NetCDFWriter(const ProblemSetup& a_ps):
    ps(a_ps),
    solTmp(ps.nX*ps.nY*ps.nZ,0.0)
  {
    // Fill coordinate vectors
    grid[0].resize(ps.nX,0.0);
    for (int n=0; n<ps.nX; ++n) {
      grid[0][n] = ps.xLo + (static_cast<double>(n) + 0.5)*ps.dx;
    }
    
    grid[1].resize(ps.nY,0.0);
    for (int n=0; n<ps.nY; ++n) {
      grid[1][n] = ps.yLo + (static_cast<double>(n) + 0.5)*ps.dy;
    }
#if (SPACE_DIM == 3)
    grid[2].resize(ps.nZ,0.0);
    for (int n=0; n<ps.nZ; ++n) {
      grid[2][n] = ps.zLo + (static_cast<double>(n) + 0.5)*ps.dz;
    }
#endif
  }

  std::string NetCDFWriter::FileName(int step)
  {
    std::ostringstream os;
    os << ps.baseName << std::setw(6) << std::setfill('0') << step << ".nc";
    return os.str();
  }

  void NetCDFWriter::write(CellDataView U,int step,double time)
  {
    // Create filename and open
    std::string fileName = FileName(step);
    std::printf("Writing solution file: %s for time %.3e\n",fileName.c_str(),time);

    try {
      netCDF::NcFile outFile(fileName,netCDF::NcFile::replace);
    
      // Write out time
      auto tDim = outFile.addDim("t",1);
      auto tVar = outFile.addVar("t",netCDF::ncDouble,tDim);
    
      // Write out domain info
      auto xDim = outFile.addDim("x",ps.nX);
      auto yDim = outFile.addDim("y",ps.nY);
      auto xVar = outFile.addVar("x",netCDF::ncDouble,xDim);
      auto yVar = outFile.addVar("y",netCDF::ncDouble,yDim);
#if (SPACE_DIM == 3)
      auto zDim = outFile.addDim("z",ps.nZ);
      auto zVar = outFile.addVar("z",netCDF::ncDouble,zDim);
      std::vector<netCDF::NcDim> dimVec({zDim,yDim,xDim});
#else
      std::vector<netCDF::NcDim> dimVec({yDim,xDim});
#endif

      // Write out solution variable info
      std::vector<netCDF::NcVar> solVar;
      for (int nV=0; nV<NUM_VARS; nV++) {
	solVar.emplace_back(outFile.addVar(ps.varName[nV],netCDF::ncDouble,dimVec));
      }

      // Write out time and domain variables
      xVar.putVar(grid[0].data());
      yVar.putVar(grid[1].data());
#if (SPACE_DIM == 3)
      zVar.putVar(grid[2].data());
#endif
      tVar.putVar(&time);
    
      // Write out solution variables, have to make temporary copy due to slicing
      auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),U);
      for (int nV=0; nV<NUM_VARS; nV++) {
	for (int nX=0; nX<ps.nX; nX++) {
	  for (int nY=0; nY<ps.nY; nY++) {
	    for (int nZ=0; nZ<ps.nZ; nZ++) {
	      int idx = nX + ps.nX*nY + ps.nX*ps.nY*nZ;
	      solTmp[idx] = h_U(KFVM_D_DECL(nX + ps.rad,nY + ps.rad,nZ + ps.rad),nV);
	    }
	  }
	}
	// Write out variables
	solVar[nV].putVar(solTmp.data());
      }
    } catch (netCDF::exceptions::NcException& e) {
      std::cout << "Error writing netCDF file:\n" << e.what() << std::endl;
    }
  }
  
} // end namespace KFVM
