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
      int nX_g,nY_g,nZ_g;
      std::vector<Real> xCoord,yCoord,zCoord;
      std::string prefix;
      std::string filename_h5;
      std::string filename_xmf;

      void writeAttrHydro(std::ofstream&,bool);
      void writeAttrMHD_GLM(std::ofstream&,bool);
      void writeAttrSRHydro(std::ofstream&,bool);
      
      void writeAttributeScalar(std::ofstream&,const char*);
      void writeAttributeVector(std::ofstream&,const char*,const char*,const char*,const char*);
      
      void writeXML(int,Real,bool);
      void writePlotPDI(ConsDataView,AuxDataView,CellDataView,int,Real);
      void writeCkptPDI(ConsDataView,CellDataView,int,Real,Real);
    public:
      WriterPDI(ProblemSetup&,const Geometry&);
      void writePlot(ConsDataView,AuxDataView,CellDataView,int,Real);
      void writeCkpt(ConsDataView,CellDataView,int,Real,Real);
      void readCkpt(ConsDataView,CellDataView,int&,Real&,Real&);
    };

  } // end namespace IO

} // end namespace KFVM

#endif
