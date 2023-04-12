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
      writeAttributeScalar(ofs,"dens");
      writeAttributeVector(ofs,"mom","momx","momy","momz");
      if (eqType == EquationType::MHD_GLM) {
        writeAttributeVector(ofs,"mag","magx","magy","magz");
      }
      writeAttributeScalar(ofs,"etot");
      writeAttributeVector(ofs,"vel","velx","vely","velz");
      writeAttributeScalar(ofs,"eint");
      writeAttributeScalar(ofs,"pres");
      if (eqType == EquationType::MHD_GLM) {
        writeAttributeScalar(ofs,"prsg");
        writeAttributeScalar(ofs,"prsb");
      }
      writeAttributeScalar(ofs,"weno");

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
          << "      <Geometry GeometryType=\"Origin_DxDy\">\n"
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
          << "      <Geometry GeometryType=\"Origin_DxDyDz\">\n"
          << "        <DataItem Name=\"Origin\" Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << ps.xLo << " " << ps.yLo << " " << ps.zLo << "\n"
          << "        </DataItem>\n"
          << "        <DataItem Name=\"Spacing\" Dimensions=\"3\" NumberType=\"Double\" Precision=\"8\" Format=\"XML\">\n"
          << "          " << geom.dx << " " << geom.dy << " " << geom.dz << "\n"
          << "        </DataItem>\n      </Geometry>\n"
          << std::endl;
    }
    
    void WriterPDI::writeAttributeScalar(std::ofstream& ofs,const char *varName)
    {
      ofs << "      <Attribute Name=\"" << varName << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
          << "        <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
#if (SPACE_DIM == 2)
          << ps.nY << " " << ps.nX << "\">\n"
#else
          << ps.nZ << " " << ps.nY << " " << ps.nX << "\">\n"
#endif
          << "          " << filename_h5 << ":/" << varName << "\n"
          << "        </DataItem>\n      </Attribute>\n"
          << std::endl;
    }
    
    void WriterPDI::writeAttributeVector(std::ofstream& ofs,const char *vecName,
                                         const char *vecX,const char *vecY,const char *vecZ)
    {
      if (SPACE_DIM == 2) {
        writeAttributeVector2D(ofs,vecName,vecX,vecY,vecZ);
      } else {
        writeAttributeVector3D(ofs,vecName,vecX,vecY,vecZ);
      }
    }

    // **Note**
    // This function permutes the vector components to undo a bug in the xdmf library
    // This hack should be temporary, but until Paraview and Visit work or xdmf is fixed
    // it will stay...
    void WriterPDI::writeAttributeVector2D(std::ofstream& ofs,const char *vecName,
                                           const char *vecX,const char *vecY,const char *vecZ)
    {
      ofs << "      <Attribute Name=\"" << vecName << "\" AttributeType=\"Vector\" Center=\"Cell\">\n"
          << "        <DataItem ItemType=\"Function\" Function=\"JOIN($0, $1, $2)\" Dimensions=\""
          << ps.nY << " " << ps.nX << " 3\">\n"
        // x-component (permutes to the Z component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecZ << "\n"
          << "          </DataItem>\n"
        // y-component (permutes to the x component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecX << "\n"
          << "          </DataItem>\n"
        // z-component (permutes to the y component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecY << "\n"
          << "          </DataItem>\n"
        
          << "        </DataItem>\n      </Attribute>\n"
          << std::endl;
    }

    // **Note**
    // The 3D case does not have the weird permutation bug, this function does what you'd hope.
    void WriterPDI::writeAttributeVector3D(std::ofstream& ofs,const char *vecName,
                                           const char *vecX,const char *vecY,const char *vecZ)
    {
      ofs << "      <Attribute Name=\"" << vecName << "\" AttributeType=\"Vector\" Center=\"Cell\">\n"
          << "        <DataItem ItemType=\"Function\" Function=\"JOIN($0, $1, $2)\" Dimensions=\""
          << ps.nZ << " " << ps.nY << " " << ps.nX << " 3\">\n"
        // x-component (permutes to the  component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nZ << " " << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecX << "\n"
          << "          </DataItem>\n"
        // y-component (permutes to the  component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nZ << " " << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecY << "\n"
          << "          </DataItem>\n"
        // z-component (permutes to the  component)
          << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nZ << " " << ps.nY << " " << ps.nX << "\">\n"
          << "            " << filename_h5 << ":/" << vecZ << "\n"
          << "          </DataItem>\n"
        
          << "        </DataItem>\n      </Attribute>\n"
          << std::endl;
    }
    
  } // end namespace IO
  
} // end namespace KFVM
