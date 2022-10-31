#ifndef NETCDFWRITER_H_
#define NETCDFWRITER_H_

#include <string>
#include <array>
#include <vector>

#include <netcdf>

#include <Definitions.H>

#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {

  class NetCDFWriter
  {
    const ProblemSetup& ps;
    std::array<std::vector<double>,SPACE_DIM> grid,gridHalo;
    std::vector<double> solTmp,solTmpHalo;

    std::string FileName(int);
    std::string FileNameHalo(int);
  public:
    NetCDFWriter(const ProblemSetup&);
    void write(CellDataView,AuxDataView,int,double);
    void writeHalo(CellDataView,int,double);
  };

}
#endif
