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
      weno_host("weno_host",KFVM_D_DECL(ps.nX,ps.nY,ps.nZ)),
      xCoord(ps.nX + 1,Real(0.0)),
      yCoord(ps.nY + 1,Real(0.0)),
      zCoord(ps.nZ + 1,Real(0.0))
    {
      // Generate base filename and make directories as needed
      std::ostringstream oss;
      oss << ps.dataDir << "/R" << ps.rad
          << "_NX" << ps.nX*ps.nbX
          << "_NY" << ps.nY*ps.nbY;
#if (SPACE_DIM == 3)
      oss << "_NZ" << ps.nZ*ps.nbZ;
#endif
      oss << "/Rank" << ps.layoutMPI.rank << "/";
      std::filesystem::create_directories(oss.str());
      prefix = std::string(oss.str());

      // Fill coordinate arrays (including z even in 2D)
      // Note that these are nodal, hence one longer than number of cells
      for (int n=0; n<=ps.nX; n++) {
        xCoord[n] = n*geom.dx + geom.xLo;
      }
      for (int n=0; n<=ps.nY; n++) {
        yCoord[n] = n*geom.dy + geom.yLo;
      }
      Real dz = (SPACE_DIM == 2 ? geom.dmin : geom.dz);
      for (int n=0; n<=ps.nZ; n++) {
        zCoord[n] = n*dz + geom.zLo;
      }

      // Gather up solution metadata and give it to PDI
      // These are all size 3, even in 2D
      int ngZ = (SPACE_DIM==2 ? 0 : (int) ps.rad);
      int siZ = (SPACE_DIM==2 ? 0 : (int) ps.rad);
      std::array<int,3> ncell = {(int) ps.nX,(int) ps.nY,(int) ps.nZ};
      std::array<int,3> nghost = {(int) ps.rad,(int) ps.rad,ngZ};
      std::array<int,3> start_idx = {(int) ps.rad,(int) ps.rad,siZ};
      
      Real time = 0.0;
      int time_step = 0;

      PDI_multi_expose("init_pdi",
                       "ncell",(void*) ncell.data(),PDI_OUT,
                       "nghost",(void*) nghost.data(),PDI_OUT,
                       "start_idx",(void*) start_idx.data(),PDI_OUT,
                       "xcoord",(void*) xCoord.data(),PDI_OUT,
                       "ycoord",(void*) yCoord.data(),PDI_OUT,
                       "zcoord",(void*) zCoord.data(),PDI_OUT,
                       "gamma",(void*) &ps.fluidProp.gamma,PDI_OUT,
                       "time",(void*) &time,PDI_OUT,
                       "time_step",(void*) &time_step,PDI_OUT,
                       NULL);

      PDI_event("init_pdi");
    }

    void WriterPDI::write(ConsDataView U,AuxDataView V,CellDataView weno,
                          int step,Real time)
    {
      // Form filenames
      std::ostringstream oss;
      oss << ps.baseName << "_" << std::setw(7) << std::setfill('0') << step;
      filename_xmf = oss.str() + ".xmf";
      filename_h5 = oss.str() + ".h5";
      
      writeXML(step,time);
      writePDI(U,V,weno,step,time);
    }

    void WriterPDI::writePDI(ConsDataView U,AuxDataView V,CellDataView weno,
                             int step,Real time)
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

    void WriterPDI::writeXML(int step,Real time)
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
      ofs << "    <Grid Name=\"Structured Grid\" GridType=\"Uniform\">\n"
          << "      <Time Value=\"" << time << "\" />\n"
          << "      <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\""
          << (ps.nZ + 1) << " " << (ps.nY + 1) << " " << (ps.nX + 1) << "\"/>\n"
          << "      <Geometry GeometryType=\"VxVyVz\">\n"
          << "        <DataItem Name=\"Vx\" Dimensions=\"" << (ps.nX + 1) << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
          << "          " << filename_h5 << ":/xcoord\n"
          << "        </DataItem>\n"
          << "        <DataItem Name=\"Vy\" Dimensions=\"" << (ps.nY + 1) << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
          << "          " << filename_h5 << ":/ycoord\n"
          << "        </DataItem>\n"
          << "        <DataItem Name=\"Vz\" Dimensions=\"" << (ps.nZ + 1) << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
          << "          " << filename_h5 << ":/zcoord\n"
          << "        </DataItem>\n      </Geometry>\n"
          << std::endl;

      // Write all data fields as attributes
      switch (eqType) {
      case EquationType::MHD_GLM:
        writeAttrMHD_GLM(ofs);
        break;
      case EquationType::SRHydro:
        writeAttrSRHydro(ofs);
        break;
      default:
        writeAttrHydro(ofs);
      }
      writeAttributeScalar(ofs,"weno");

      // Close grid, domain, Xdmf
      ofs << "    </Grid>\n  </Domain>\n</Xdmf>" << std::endl;

      // Close file
      ofs.close();
    }

    void WriterPDI::writeAttrHydro(std::ofstream& ofs)
    {    
      writeAttributeScalar(ofs,"dens");
      writeAttributeVector(ofs,"mom","momx","momy","momz");
      writeAttributeScalar(ofs,"etot");
      writeAttributeVector(ofs,"vel","velx","vely","velz");
      writeAttributeScalar(ofs,"eint");
      writeAttributeScalar(ofs,"pres");
    }

    void WriterPDI::writeAttrMHD_GLM(std::ofstream& ofs)
    {
      writeAttributeScalar(ofs,"dens");
      writeAttributeVector(ofs,"mom","momx","momy","momz");
      writeAttributeVector(ofs,"mag","magx","magy","magz");
      writeAttributeScalar(ofs,"etot");
      writeAttributeVector(ofs,"vel","velx","vely","velz");
      writeAttributeScalar(ofs,"eint");
      writeAttributeScalar(ofs,"pres");
      writeAttributeScalar(ofs,"prsg");
      writeAttributeScalar(ofs,"prsb");
    }

    void WriterPDI::writeAttrSRHydro(std::ofstream& ofs)
    {
      writeAttributeScalar(ofs,"dens");
      writeAttributeVector(ofs,"mom","momx","momy","momz");
      writeAttributeScalar(ofs,"tau");
      writeAttributeScalar(ofs,"rho");
      writeAttributeVector(ofs,"vel","velx","vely","velz");
      writeAttributeScalar(ofs,"lorz");
      writeAttributeScalar(ofs,"pres");
    }
    
    void WriterPDI::writeAttributeScalar(std::ofstream& ofs,const char *varName)
    {
      ofs << "      <Attribute Name=\"" << varName << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
          << "        <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" Dimensions=\""
          << ps.nZ << " " << ps.nY << " " << ps.nX << "\">\n"
          << "          " << filename_h5 << ":/" << varName << "\n"
          << "        </DataItem>\n      </Attribute>\n"
          << std::endl;
    }
    
    void WriterPDI::writeAttributeVector(std::ofstream& ofs,const char *vecName,
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
