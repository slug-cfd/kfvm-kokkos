#ifndef SOLUTIONWRITER_H_
#define SOLUTIONWRITER_H_

#include <string>
#include <array>
#include <vector>

#include <hdf5.h>

#include "SimVar.H"
#include "Types.H"
#include "ProblemSetup.H"

namespace KFVM {

  class SolutionWriter
  {
    const ProblemSetup& ps;
    const std::array<hsize_t,SPACE_DIM> gridSize;
    const std::array<double,2*SPACE_DIM> extent;
    std::array<std::vector<double>,SPACE_DIM> grid;
    std::vector<double> solTmp;

    std::string FileName(int);
  public:
    SolutionWriter(const ProblemSetup& ps);
    void write(CellDataView,int,double);
  };

}
#endif
