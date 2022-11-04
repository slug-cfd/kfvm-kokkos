#include <cstddef>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>

#include <Kokkos_Core.hpp>

#include <netcdf>

#include <Definitions.H>

#include "Dimension.H"
#include "Geometry.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "numeric/Numeric.H"
#include "NetCDFWriter.H"

namespace KFVM {

  NetCDFWriter::NetCDFWriter(const ProblemSetup& a_ps):
    ps(a_ps),
    solTmp(ps.nX*ps.nY*ps.nZ,0.0),
    solTmpHalo((ps.nX + 2*ps.rad)*(ps.nY + 2*ps.rad)*(ps.nZ + 2*ps.rad),0.0)
  {
    // Fill coordinate vectors
    grid[0].resize(ps.nX,0.0);
    gridHalo[0].resize(ps.nX + 2*ps.rad,0.0);
    for (idx_t n=0; n<ps.nX; ++n) {
      grid[0][n] = ps.xLo + (static_cast<double>(n) + 0.5)*ps.dx;
      gridHalo[0][n + ps.rad] = grid[0][n];
    }
    
    grid[1].resize(ps.nY,0.0);
    gridHalo[1].resize(ps.nY + 2*ps.rad,0.0);
    for (idx_t n=0; n<ps.nY; ++n) {
      grid[1][n] = ps.yLo + (static_cast<double>(n) + 0.5)*ps.dy;
      gridHalo[1][n + ps.rad] = grid[1][n];
    }
    
#if (SPACE_DIM == 3)
    grid[2].resize(ps.nZ,0.0);
    gridHalo[2].resize(ps.nZ + 2*ps.rad,0.0);
    for (idx_t n=0; n<ps.nZ; ++n) {
      grid[2][n] = ps.zLo + (static_cast<double>(n) + 0.5)*ps.dz;
      gridHalo[2][n + ps.rad] = grid[2][n];
    }
#endif

    // add in halo positions
    for (idx_t nH=1; nH<=ps.rad; nH++) {
      gridHalo[0][ps.rad - nH]             = gridHalo[0][ps.rad] - nH*ps.dx;
      gridHalo[0][ps.rad + ps.nX + nH - 1] = gridHalo[0][ps.rad + ps.nX - 1] + nH*ps.dx;
      gridHalo[1][ps.rad - nH]             = gridHalo[1][ps.rad] - nH*ps.dy;
      gridHalo[1][ps.rad + ps.nY + nH - 1] = gridHalo[1][ps.rad + ps.nY - 1] + nH*ps.dy;
#if (SPACE_DIM == 3)
      gridHalo[2][ps.rad - nH]             = gridHalo[2][ps.rad] - nH*ps.dz;
      gridHalo[2][ps.rad + ps.nZ + nH - 1] = gridHalo[2][ps.rad + ps.nZ - 1] + nH*ps.dz;
#endif
    }
  }

  std::string NetCDFWriter::FileName(int step)
  {
    std::ostringstream os;
    os << ps.baseName << std::setw(6) << std::setfill('0') << step << ".nc";
    return os.str();
  }

  std::string NetCDFWriter::FileNameHalo(int step)
  {
    std::ostringstream os;
    os << ps.baseName << "Halo" << std::setw(6) << std::setfill('0') << step << ".nc";
    return os.str();
  }

  void NetCDFWriter::write(CellDataView U,AuxDataView V,int step,double time)
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
      for (int nV=0; nV<NUM_AUX; nV++) {
	solVar.emplace_back(outFile.addVar(ps.auxVarName[nV],netCDF::ncDouble,dimVec));
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
      for (idx_t nV=0; nV<NUM_VARS; nV++) {
	for (idx_t nX=0; nX<ps.nX; nX++) {
	  for (idx_t nY=0; nY<ps.nY; nY++) {
	    for (idx_t nZ=0; nZ<ps.nZ; nZ++) {
	      idx_t idx = nX + ps.nX*nY + ps.nX*ps.nY*nZ;
	      solTmp[idx] = h_U(KFVM_D_DECL(nX + ps.rad,nY + ps.rad,nZ + ps.rad),nV);
	    }
	  }
	}
	// Write out variables
	solVar[nV].putVar(solTmp.data());
      }
      
      // Write out auxiliary variables, have to make temporary copy due to slicing
      auto h_V = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),V);
      for (idx_t nV=0; nV<NUM_AUX; nV++) {
	for (idx_t nX=0; nX<ps.nX; nX++) {
	  for (idx_t nY=0; nY<ps.nY; nY++) {
	    for (idx_t nZ=0; nZ<ps.nZ; nZ++) {
	      idx_t idx = nX + ps.nX*nY + ps.nX*ps.nY*nZ;
	      solTmp[idx] = h_V(KFVM_D_DECL(nX,nY,nZ),nV);
	    }
	  }
	}
	// Write out variables
	solVar[nV + NUM_VARS].putVar(solTmp.data());
      }
    } catch (netCDF::exceptions::NcException& e) {
      std::cout << "Error writing netCDF file:\n" << e.what() << std::endl;
    }
  }

  void NetCDFWriter::writeHalo(CellDataView U,int step,double time)
  {
    // Create filename and open
    std::string fileName = FileNameHalo(step);
    std::printf("Writing solution file: %s for time %.3e\n",fileName.c_str(),time);

    try {
      netCDF::NcFile outFile(fileName,netCDF::NcFile::replace);
    
      // Write out time
      auto tDim = outFile.addDim("t",1);
      auto tVar = outFile.addVar("t",netCDF::ncDouble,tDim);
    
      // Write out domain info
      auto xDim = outFile.addDim("x",ps.nX + 2*ps.rad);
      auto yDim = outFile.addDim("y",ps.nY + 2*ps.rad);
      auto xVar = outFile.addVar("x",netCDF::ncDouble,xDim);
      auto yVar = outFile.addVar("y",netCDF::ncDouble,yDim);
#if (SPACE_DIM == 3)
      auto zDim = outFile.addDim("z",ps.nZ + 2*ps.rad);
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
      xVar.putVar(gridHalo[0].data());
      yVar.putVar(gridHalo[1].data());
#if (SPACE_DIM == 3)
      zVar.putVar(gridHalo[2].data());
#endif
      tVar.putVar(&time);
    
      // Write out solution variables, have to make temporary copy due to slicing
      auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),U);
      for (idx_t nV=0; nV<NUM_VARS; nV++) {
	for (idx_t nX=0; nX<ps.nX + 2*ps.rad; nX++) {
	  for (idx_t nY=0; nY<ps.nY + 2*ps.rad; nY++) {
#if (SPACE_DIM == 2)
            idx_t idx = nX + (ps.nX + 2*ps.rad)*nY;
            solTmp[idx] = h_U(nX,nY,nV);
#else
	    for (idx_t nZ=0; nZ<ps.nZ + 2*ps.rad; nZ++) {
	      idx_t idx = nX + (ps.nX + 2*ps.rad)*nY + (ps.nX + 2*ps.rad)*(ps.nY + 2*ps.rad)*nZ;
	      solTmp[idx] = h_U(KFVM_D_DECL(nX,nY,nZ),nV);
	    }
#endif
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
