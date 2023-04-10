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
      std::ostringstream oss;
      oss << ps.dataDir << "/R" << ps.rad
          << "_NX" << ps.nX
          << "_NY" << ps.nY;
#if (SPACE_DIM == 3)
      oss << "_NZ" << ps.nZ;
#endif
      std::filesystem::create_directories(oss.str());
      oss << "/";
      prefix = std::string(oss.str());

      // Gather up solution metadata and give it to PDI
      std::array<int,SPACE_DIM> ncell =
        {KFVM_D_DECL((int)ps.nX,(int)ps.nY,(int)ps.nZ)};
      std::array<int,SPACE_DIM> nghost =
        {KFVM_D_DECL((int)ps.rad,(int)ps.rad,(int)ps.rad)};
      std::array<int,SPACE_DIM> start_idx =
        {KFVM_D_DECL((int)ps.rad,(int)ps.rad,(int)ps.rad)};
      
      Real time = 0.0;
      int time_step = 0;

      PDI_multi_expose("init_pdi",
                       "ncell",(void*) ncell.data(),PDI_OUT,
                       "nghost",(void*) nghost.data(),PDI_OUT,
                       "start_idx",(void*) start_idx.data(),PDI_OUT,
                       "gamma",(void*) &ps.fluidProp.gamma,PDI_OUT,
                       "time",(void*) &time,PDI_OUT,
                       "time_step",(void*) &time_step,PDI_OUT,
                       NULL);

      PDI_event("init_pdi");
    }

    void WriterPDI::write(ConsDataView U,AuxDataView V,CellDataView weno,
                          int step,double time)
    {
      // Form filenames
      std::ostringstream oss;
      oss << ps.baseName << "_" << std::setw(7) << std::setfill('0') << step;
      filename_xmf = oss.str() + ".xmf";
      filename_h5 = oss.str() + ".h5";
      
      writeXML(step);
      writePDI(U,V,weno,step,time);
    }

    void WriterPDI::writePDI(ConsDataView U,AuxDataView V,CellDataView weno,
                             int step,double time)
    {
      std::string filename = prefix + filename_h5;
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

    void WriterPDI::writeXML(int step)
    {
      std::string filename = prefix + filename_xmf;
      std::cout << "Writing file: " << filename << std::endl;

      // Create Xdmf file
      std::ofstream ofs(filename,std::ios::trunc);

      // Write header, open Xdmf and domain
      ofs << "<?xml version=\"1.0\" ?>\n"
          << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
          << "<Xdmf Version=\"2.0\">\n"
          << "  <Domain>"
          << std::endl;

      // Create grid, topology, and geometry
      if (SPACE_DIM == 2) {
        writeGrid2D(ofs);
      } else {
        writeGrid3D(ofs);
      }

      // Write all data fields as attributes
      for (int nV=0; nV<NUM_VARS; nV++) {
        writeAttribute(ofs,ps.varName[nV]);
      }
      for (int nV=0; nV<NUM_AUX; nV++) {
        writeAttribute(ofs,ps.auxVarName[nV]);
      }
      writeAttribute(ofs,std::string("weno"));

      // Close grid, domain, Xdmf
      ofs << "    </Grid>\n  </Domain>\n</Xdmf>" << std::endl;

      // Close file
      ofs.close();
    }

    void WriterPDI::writeGrid2D(std::ofstream& ofs)
    {
      ofs << "    <Grid Name=\"Structured Grid\" GridType=\"Uniform\">\n"
          << "      <Topology TopologyType=\"2DCoRectMesh\" NumberOfElements=\""
          << (ps.nY + 1) << " " << (ps.nX + 1) << "\"/>\n"
          << "      <Geometry GeometryType=\"Origin_DXDY\">\n"
          << "        <DataItem Name=\"Origin\" Dimensions=\"2\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << ps.xLo << " " << ps.yLo << " " << "\n"
          << "        </DataItem>\n"
          << "        <DataItem Name=\"Spacing\" Dimensions=\"2\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << geom.dx << " " << geom.dy << "\n"
          << "        </DataItem>\n      </Geometry>\n"
          << std::endl;
    }

    void WriterPDI::writeGrid3D(std::ofstream& ofs)
    {
      ofs << "    <Grid Name=\"Structured Grid\" GridType=\"Uniform\">\n"
          << "      <Topology TopologyType=\"3DCoRectMesh\" NumberOfElements=\""
          << (ps.nZ + 1) << " " << (ps.nY + 1) << " " << (ps.nX + 1) << "\"/>\n"
          << "      <Geometry GeometryType=\"Origin_DXDYDZ\">\n"
          << "        <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << ps.xLo << " " << ps.yLo << " " << ps.zLo << "\n"
          << "        </DataItem>\n"
          << "        <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << geom.dx << " " << geom.dy << " " << geom.dz << "\n"
          << "        </DataItem>\n      </Geometry>\n"
          << std::endl;
    }
    
    void WriterPDI::writeAttribute(std::ofstream& ofs,const std::string& varName)
    {
      ofs << "      <Attribute Name=\"" << varName << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
          << "        <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
#if (SPACE_DIM == 3)        
          << ps.nZ << " "
#endif
          << ps.nY << " "
          << ps.nX << "\">\n"
          << "          " << filename_h5 << ":/" << varName << "\n"
          << "        </DataItem>\n      </Attribute>\n"
          << std::endl;
    }
    
  } // end namespace IO
  
} // end namespace KFVM
