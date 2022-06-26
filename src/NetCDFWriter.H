#ifndef NETCDFWRITER_H_
#define NETCDFWRITER_H_

#include <string>
#include <array>
#include <vector>

#include <netcdf>

#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {

  class NetCDFWriter
  {
    const ProblemSetup& ps;
    const std::array<idx_t,SPACE_DIM> gridSize;
    std::array<std::vector<double>,SPACE_DIM> grid;
    std::vector<double> solTmp;

    std::string FileName(idx_t);
  public:
    NetCDFWriter(const ProblemSetup& ps);
    void write(CellDataView,idx_t,double);
  };

}
#endif
