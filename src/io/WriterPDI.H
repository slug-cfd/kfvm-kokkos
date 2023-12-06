#ifndef WRITERPDI_H_
#define WRITERPDI_H_

#include <string>
#include <vector>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "../Geometry.H"
#include "../ProblemSetup.H"
#include "../Types.H"
#include "../physics/SimVar.H"

namespace KFVM {

namespace IO {

class WriterPDI {
  ProblemSetup &ps;
  const Geometry geom;
  HostConsDataView U_host;
  HostWenoFlagView weno_host;
  int nX_g, nY_g, nZ_g;
  std::vector<Real> xCoord, yCoord, zCoord;
  std::string prefix;
  std::string filename_h5;
  std::string filename_xmf;
  int stats_row_num;

  void writeAttrHydro(std::ofstream &, bool);
  void writeAttrLinAdv(std::ofstream &, bool);
  void writeAttrMHD_GLM(std::ofstream &, bool);
  void writeAttrMHD_8W(std::ofstream &, bool);
  void writeAttrSRHydro(std::ofstream &, bool);

  void writeAttributeScalar(std::ofstream &, const char *);
  void writeAttributeVector(std::ofstream &, const char *, const char *, const char *,
                            const char *);

  void writeXML(int, Real, bool);
  void writePlotPDI(ConsDataView, AuxDataView, WenoFlagView, int, Real);
  void writeCkptPDI(ConsDataView, WenoFlagView, int, Real, Real);

 public:
  WriterPDI(ProblemSetup &, const Geometry &);
  void writeFlowStats(const Physics::FlowStatsArray &, Real);
  void writePlot(ConsDataView, AuxDataView, WenoFlagView, int, Real);
  void writeCkpt(ConsDataView, WenoFlagView, int, Real, Real);
  void readCkpt(ConsDataView, WenoFlagView, int &, Real &, Real &);
};

} // end namespace IO

} // end namespace KFVM

#endif
