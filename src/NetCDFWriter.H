#ifndef NETCDFWRITER_H_
#define NETCDFWRITER_H_

#include <string>
#include <array>
#include <vector>

#include <netcdf>

#include <Definitions.H>
#include <SimVar.H>
#include <Types.H>
#include <ProblemSetup.H>

namespace KFVM {

  class NetCDFWriter
  {
    const ProblemSetup& ps;
    const std::array<int,SPACE_DIM> gridSize;
    std::array<std::vector<double>,SPACE_DIM> grid;
    std::vector<double> solTmp;

    std::string FileName(int);
  public:
    NetCDFWriter(const ProblemSetup& ps);
    void write(CellDataView,int,double);
  };

}
#endif
