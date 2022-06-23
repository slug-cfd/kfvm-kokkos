#include <H5public.h>
#include <Kokkos_CopyViews.hpp>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>

#include <hdf5.h>

#include "Dimension.H"
#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"
#include "SolutionWriter.H"

namespace KFVM {

  SolutionWriter::SolutionWriter(const ProblemSetup& a_ps):
    ps(a_ps),
    gridSize({KFVM_D_DECL(hsize_t(ps.nX),hsize_t(ps.nY),hsize_t(ps.nZ))}),
#if (SPACE_DIM == 2)
    extent({ps.xLo,ps.xUp,ps.yLo,ps.yUp}),
#else
    extent({ps.xLo,ps.xUp,ps.yLo,ps.yUp,ps.zLo,ps.zUp}),
#endif
    solTmp(ps.nX*ps.nY*ps.nZ,0.0)
  {
    // Fill coordinate vectors
    grid[0].resize(ps.nX,0.0);
    for (std::size_t n=0; n<ps.nX; ++n) {
      grid[0][n] = ps.xLo + (static_cast<double>(n) + 0.5)*ps.dx;
    }
    
    grid[1].resize(ps.nY,0.0);
    for (std::size_t n=0; n<ps.nY; ++n) {
      grid[1][n] = ps.yLo + (static_cast<double>(n) + 0.5)*ps.dy;
    }
#if (SPACE_DIM == 3)
    grid[2].resize(ps.nZ,0.0);
    for (std::size_t n=0; n<ps.nZ; ++n) {
      grid[2][n] = ps.zLo + (static_cast<double>(n) + 0.5)*ps.dz;
    }
#endif
  }

  std::string SolutionWriter::FileName(int step)
  {
    std::ostringstream os;
    os << ps.baseName << std::setw(6) << std::setfill('0') << step << ".h5";
    return os.str();
  }

  void SolutionWriter::write(CellDataView U,int step,double time)
  {
    // Create filename and open
    std::string fileName = FileName(step);
    std::printf("Writing solution file: %s for time %.3e\n",fileName.c_str(),time);
    hid_t file_id = H5Fcreate(fileName.c_str(),H5F_ACC_TRUNC,H5P_DEFAULT,H5P_DEFAULT);
    // Write out time
    hsize_t one = 1;
    hid_t dataspace_t = H5Screate_simple(1,&one,&one);
    hid_t dataset_t = H5Dcreate(file_id,"time",H5T_NATIVE_DOUBLE,
				dataspace_t,H5P_DEFAULT,
				H5P_DEFAULT,H5P_DEFAULT);
    H5Dwrite(dataset_t,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,H5P_DEFAULT,&time);
    H5Dclose(dataset_t);
    
    // Write out domain info
    hsize_t nExt = 2*SPACE_DIM;
    hid_t dataspace_ext = H5Screate_simple(1,&nExt,&nExt);
    hid_t dataset_ext = H5Dcreate(file_id,"ext",H5T_NATIVE_DOUBLE,
				  dataspace_ext,H5P_DEFAULT,
				  H5P_DEFAULT,H5P_DEFAULT);
    H5Dwrite(dataset_ext,H5T_NATIVE_DOUBLE,
	     H5S_ALL,H5S_ALL,H5P_DEFAULT,extent.data());
    H5Dclose(dataset_ext);

    for (int nD=0; nD<SPACE_DIM; nD++) {
      hid_t dataspace_gr = H5Screate_simple(1,&gridSize[nD],&gridSize[nD]);
      hid_t dataset_gr = H5Dcreate(file_id,
				   nD==0 ? "x" : (nD==1 ? "y" : "z"),
				   H5T_NATIVE_DOUBLE,
				   dataspace_gr,H5P_DEFAULT,
				   H5P_DEFAULT,H5P_DEFAULT);
      H5Dwrite(dataset_gr,H5T_NATIVE_DOUBLE,H5S_ALL,
	       H5S_ALL,H5P_DEFAULT,grid[nD].data());
      H5Dclose(dataset_gr);
    }
    
    // Write out solution variables, have to make temp copy due to slicing
    hid_t dataspace_id = H5Screate_simple(SPACE_DIM,
					  gridSize.data(),gridSize.data());
    // Get host view and make a temporary copy
    auto h_U = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),U);
    for(int nV=0; nV<NUM_VARS; ++nV) {
      for (int nX=0; nX<ps.nX; ++nX) {
        for (int nY=0; nY<ps.nY; ++nY) {
	  for (int nZ=0; nZ<ps.nZ; ++nZ) {
	    std::size_t idx = nZ + ps.nZ*nY + ps.nZ*ps.nY*nX;
	    //std::size_t idx = nX + ps.nX*nY + ps.nX*ps.nY*nZ;
	    solTmp[idx] = h_U(KFVM_D_DECL(nX + ps.rad,nY + ps.rad,nZ + ps.rad),nV);
	  }
	}
      }
      
      // Write out variable
      hid_t dataset_id = H5Dcreate(file_id,ps.varName[nV].c_str(),
				   H5T_NATIVE_DOUBLE,dataspace_id,
				   H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT);
      H5Dwrite(dataset_id,H5T_NATIVE_DOUBLE,H5S_ALL,H5S_ALL,
	       H5P_DEFAULT,solTmp.data());
      H5Dclose(dataset_id);
    }
    // Close file and finalize
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
  }
  
}
