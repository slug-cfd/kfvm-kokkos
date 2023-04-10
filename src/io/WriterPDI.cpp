#include <iostream>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <string>
#include <filesystem>

#include <pdi.h>

#include "WriterPDI.H"

namespace KFVM {
  
  namespace IO {

    WriterPDI::WriterPDI(ProblemSetup& ps_,const Geometry& geom_):
      ps(ps_),
      geom(geom_),
      U_host("U_host",KFVM_D_DECL(ps.nX + 2*ps.rad,
                                  ps.nY + 2*ps.rad,
                                  ps.nZ + 2*ps.rad)),
      V_host("V_host",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
      weno_host("weno_host",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ))
    {
      // Generate base filename and make directories as needed
      std::ostringstream os;
      os << ps.dataDir << "/R" << ps.rad << "_NX" << ps.nX << "_NY" << ps.nY;
#if (SPACE_DIM == 3)
      os << "_NZ" << ps.nZ;
#endif
      std::filesystem::create_directories(os.str());
      os << "/" << ps.baseName;
      prefix = std::string(os.str());

      // Gather up solution metadata and give it to PDI
      std::array<int,SPACE_DIM> ncell =
        {KFVM_D_DECL((int)ps.nX,(int)ps.nY,(int)ps.nZ)};
      std::array<int,SPACE_DIM> nghost =
        {KFVM_D_DECL((int)ps.rad,(int)ps.rad,(int)ps.rad)};
      std::array<int,SPACE_DIM> start_idx =
        {KFVM_D_DECL((int)ps.rad,(int)ps.rad,(int)ps.rad)};
      std::array<Real,SPACE_DIM> origin_coord =
        {KFVM_D_DECL(Real(0.0),Real(0.0),Real(0.0))};
      std::array<Real,SPACE_DIM> grid_delta =
        {KFVM_D_DECL(geom.dx,geom.dy,geom.dz)};
      
      Real time = 0.0;
      int time_step = 0;

      PDI_multi_expose("init_pdi",
                       "ncell",(void*) ncell.data(),PDI_OUT,
                       "nghost",(void*) nghost.data(),PDI_OUT,
                       "start_idx",(void*) start_idx.data(),PDI_OUT,
                       "origin_coord",(void*) origin_coord.data(),PDI_OUT,
                       "grid_delta",(void*) grid_delta.data(),PDI_OUT,
                       "gamma",(void*) &ps.fluidProp.gamma,PDI_OUT,
                       "time",(void*) &time,PDI_OUT,
                       "time_step",(void*) &time_step,PDI_OUT,
                       NULL);

      PDI_event("init_pdi");
    }

    void WriterPDI::writePDI(ConsDataView U,AuxDataView V,CellDataView weno,int step,double time)
    {
      // Form filename
      std::ostringstream os;
      os << prefix << "_" << std::setw(7) << std::setfill('0') << step << ".h5";
      std::string filename = os.str();
      int filename_size = filename.size();
      std::cout << "Writing file: " << filename << std::endl;

      // Copy data from GPU to host (no-op if already on host)
      Kokkos::deep_copy(U_host,U);
      Kokkos::deep_copy(V_host,V);
      Kokkos::deep_copy(weno_host,weno);

      PDI_multi_expose("write_data",
                       "filename_size",(void*) &filename_size, PDI_OUT,
                       "filename",(void*) filename.c_str(), PDI_OUT,
                       "time_step",(void*) &step, PDI_OUT,
                       "time",(void*) &time, PDI_OUT,
                       "cons_field",(void*) U_host.data(), PDI_OUT,
                       "aux_field",(void*) V_host.data(), PDI_OUT,
                       "weno_field",(void*) weno_host.data(), PDI_OUT,
                       NULL);
    }

    void WriterPDI::write(ConsDataView U,AuxDataView V,CellDataView weno,int step,double time)
    {
      writePDI(U,V,weno,step,time);
    }
    
  } // end namespace IO
  
} // end namespace KFVM
