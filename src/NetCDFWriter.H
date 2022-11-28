#ifndef NETCDFWRITER_H_
#define NETCDFWRITER_H_

#include <string>
#include <array>
#include <vector>

#include <netcdf>

#include <Definitions.H>

#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {

  class NetCDFWriter
  {
    const ProblemSetup& ps;
    std::string baseName;
    std::array<std::vector<double>,SPACE_DIM> grid,gridHalo;
    std::vector<double> solTmp,solTmpHalo;

    std::string FileName(int);
    std::string FileNameHalo(int);
  public:
    NetCDFWriter(const ProblemSetup&);
    void write(ConsDataView,AuxDataView,CellDataView,int,double);
    void writeHalo(ConsDataView,int,double);
  };

}
#endif
