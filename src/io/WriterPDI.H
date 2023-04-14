#ifndef WRITERPDI_H_
#define WRITERPDI_H_

#include <vector>
#include <string>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Types.H"
#include "../ProblemSetup.H"
#include "../Geometry.H"

namespace KFVM {

  namespace IO {

    class WriterPDI
    {
      ProblemSetup& ps;
      const Geometry geom;
      HostConsDataView U_host;
      HostAuxDataView V_host;
      HostCellDataView weno_host;
      std::vector<Real> xCoord,yCoord,zCoord;
      std::string prefix;
      std::string filename_h5;
      std::string filename_xmf;

      void writeAttrHydro(std::ofstream&);
      void writeAttrMHD_GLM(std::ofstream&);
      void writeAttrSRHydro(std::ofstream&);
      void writeAttributeScalar(std::ofstream&,const char*);
      void writeAttributeVector(std::ofstream&,const char*,const char*,const char*,const char*);
      void writeXML(int,Real);
      void writePDI(ConsDataView,AuxDataView,CellDataView,int,Real);
    public:
      WriterPDI(ProblemSetup&,const Geometry&);
      void write(ConsDataView,AuxDataView,CellDataView,int,Real);
    };

  } // end namespace IO

} // end namespace KFVM

#endif
