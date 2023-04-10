#ifndef WRITERPDI_H_
#define WRITERPDI_H_

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
      std::string prefix;
      
      void writeXML();
      void writePDI(ConsDataView,AuxDataView,CellDataView,int,double);
    public:
      WriterPDI(ProblemSetup&,const Geometry&);
      void write(ConsDataView,AuxDataView,CellDataView,int,double);
    };

  } // end namespace IO

} // end namespace KFVM

#endif
