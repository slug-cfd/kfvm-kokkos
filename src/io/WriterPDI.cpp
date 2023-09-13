#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <pdi.h>

#include "../PrinterMPI.H"
#include "../Types.H"
#include "WriterPDI.H"

namespace KFVM {

namespace IO {

WriterPDI::WriterPDI(ProblemSetup &ps_, const Geometry &geom_)
    : ps(ps_), geom(geom_),
      U_host("U_host",
             KFVM_D_DECL(ps.nX + 2 * ps.rad, ps.nY + 2 * ps.rad, ps.nZ + 2 * ps.rad)),
      weno_host("weno_host", KFVM_D_DECL(ps.nX, ps.nY, ps.nZ)),
      nX_g(ps.layoutMPI.nbX * ps.nX), nY_g(ps.layoutMPI.nbY * ps.nY),
      nZ_g(ps.layoutMPI.nbZ * ps.nZ), xCoord(nX_g + 1, Real(0.0)),
      yCoord(nY_g + 1, Real(0.0)), zCoord(nZ_g + 1, Real(0.0)), stats_row_num(0) {
  // Generate base filename and make directories as needed
  std::ostringstream oss;
  oss << ps.dataDir << "/R" << ps.rad << "_NX" << ps.nX * ps.nbX << "_NY"
      << ps.nY * ps.nbY;
#if (SPACE_DIM == 3)
  oss << "_NZ" << ps.nZ * ps.nbZ;
#endif
  oss << "/";
  prefix = std::string(oss.str());
  std::filesystem::create_directories(prefix + "plot/");
  std::filesystem::create_directories(prefix + "ckpt/");

  // Fill coordinate arrays (including z even in 2D)
  // Note that these are nodal, hence one longer than number of cells
  // Note also that ps.xLo is used and *not* geom.xLo (need global low)
  for (int n = 0; n <= nX_g; n++) {
    xCoord[n] = n * geom.dx + ps.xLo_g;
  }
  for (int n = 0; n <= nY_g; n++) {
    yCoord[n] = n * geom.dy + ps.yLo_g;
  }
  Real dz = (SPACE_DIM == 2 ? geom.dmin : geom.dz);
  for (int n = 0; n <= nZ_g; n++) {
    zCoord[n] = n * dz + ps.zLo_g;
  }

  // Gather up solution metadata and give it to PDI
  // These are all size 3, even in 2D
  int ngZ = (SPACE_DIM == 2 ? 0 : int(ps.rad));
  std::array<int, 3> ncell_global = {int(nX_g), int(nY_g), int(nZ_g)};
  std::array<int, 3> ncell_local = {int(ps.nX), int(ps.nY), int(ps.nZ)};
  std::array<int, 3> nghost = {int(ps.rad), int(ps.rad), ngZ};
  std::array<int, 3> start_idx = {int(ps.layoutMPI.bxLo), int(ps.layoutMPI.byLo),
                                  int(ps.layoutMPI.bzLo)};

  PDI_multi_expose("", "ncell_g", (void *)ncell_global.data(), PDI_OUT, "ncell_l",
                   (void *)ncell_local.data(), PDI_OUT, "nghost", (void *)nghost.data(),
                   PDI_OUT, "start_idx", (void *)start_idx.data(), PDI_OUT, "xcoord",
                   (void *)xCoord.data(), PDI_OUT, "ycoord", (void *)yCoord.data(),
                   PDI_OUT, "zcoord", (void *)zCoord.data(), PDI_OUT, "gamma",
                   (void *)&ps.eosParams.gamma, PDI_OUT, "comm",
                   (void *)&ps.layoutMPI.commWorld, PDI_OUT, "stats_max_rows",
                   (void *)&ps.maxTimeSteps, PDI_OUT, NULL);
}

void WriterPDI::writeFlowStats(const Physics::FlowStatsArray &stats, Real time) {
  std::string filename = prefix + ps.baseName + "_stats.h5";
  int filename_size = filename.size();

  // Pull flow stats and time into a single array
  std::array<Real, NUM_STATS + 1> stats_row;
  stats_row[0] = time;
  for (int nS = 0; nS < NUM_STATS; nS++) {
    stats_row[nS + 1] = stats.data[nS];
  }

  PDI_multi_expose("write_flow_stats", "filename_size", (void *)&filename_size, PDI_OUT,
                   "filename", (void *)filename.c_str(), PDI_OUT, "stats_row_num",
                   (void *)&stats_row_num, PDI_OUT, "stats_row", (void *)stats_row.data(),
                   PDI_OUT, NULL);

  stats_row_num++;
}

void WriterPDI::writePlot(ConsDataView U, AuxDataView V, WenoFlagView weno, int step,
                          Real time) {
  // Form filenames
  filename_xmf = fmt::format("{}_{:07d}.xmf", ps.baseName, step);
  filename_h5 = fmt::format("{}_{:07d}.h5", ps.baseName, step);

  if (ps.layoutMPI.rank == 0) {
    writeXML(step, time, true);
  }
  writePlotPDI(U, V, weno, step, time);
}

void WriterPDI::writeCkpt(ConsDataView U, WenoFlagView weno, int step, Real time,
                          Real dt) {
  // Form filenames
  filename_xmf = fmt::format("{}_{:07d}.xmf", ps.baseName, step);
  filename_h5 = fmt::format("{}_{:07d}.h5", ps.baseName, step);

  if (ps.layoutMPI.rank == 0) {
    writeXML(step, time, false);
  }
  writeCkptPDI(U, weno, step, time, dt);
}

void WriterPDI::writePlotPDI(ConsDataView U, AuxDataView V, WenoFlagView weno, int step,
                             Real time) {
  std::string filename = prefix + "plot/" + filename_h5;
  int filename_size = filename.size();
  Print::Single(ps, "Writing file: {}\n", filename);

  // Copy data from GPU to host (no-op if already on host)
  Kokkos::deep_copy(U_host, U);
  Kokkos::deep_copy(weno_host, weno);

  PDI_multi_expose("write_plot_data", "filename_size", (void *)&filename_size, PDI_OUT,
                   "filename", (void *)filename.c_str(), PDI_OUT, "time_step",
                   (void *)&step, PDI_OUT, "time", (void *)&time, PDI_OUT, "cons_field",
                   (void *)U_host.data(), PDI_OUT, "aux_field", (void *)V.data(), PDI_OUT,
                   "weno_field", (void *)weno_host.data(), PDI_OUT, NULL);
}

void WriterPDI::writeCkptPDI(ConsDataView U, WenoFlagView weno, int step, Real time,
                             Real dt) {
  std::string filename = prefix + "ckpt/" + filename_h5;
  int filename_size = filename.size();
  Print::Single(ps, "Writing file: {}\n", filename);

  // Copy data from GPU to host (no-op if already on host)
  Kokkos::deep_copy(U_host, U);
  Kokkos::deep_copy(weno_host, weno);

  PDI_multi_expose("write_ckpt_data", "filename_size", (void *)&filename_size, PDI_OUT,
                   "filename", (void *)filename.c_str(), PDI_OUT, "time_step",
                   (void *)&step, PDI_OUT, "time", (void *)&time, PDI_OUT,
                   "time_step_size", (void *)&dt, PDI_OUT, "cons_field",
                   (void *)U_host.data(), PDI_OUT, "weno_field", (void *)weno_host.data(),
                   PDI_OUT, NULL);
}

void WriterPDI::readCkpt(ConsDataView U, WenoFlagView weno, int &step, Real &time,
                         Real &dt) {
  int filename_size = ps.restartFile.size();
  Print::Single(ps, "Restarting from: {}\n", ps.restartFile);

  PDI_multi_expose("read_ckpt_data", "restart_filename_size", (void *)&filename_size,
                   PDI_OUT, "restart_filename", (void *)ps.restartFile.c_str(), PDI_OUT,
                   "time", (void *)&time, PDI_INOUT, "time_step", (void *)&step,
                   PDI_INOUT, "time_step_size", (void *)&dt, PDI_INOUT, "cons_field",
                   (void *)U_host.data(), PDI_INOUT, "weno_field",
                   (void *)weno_host.data(), PDI_INOUT, NULL);

  // Copy data from host to GPU
  Kokkos::deep_copy(U, U_host);
  Kokkos::deep_copy(weno, weno_host);
  Kokkos::fence("WriterPDI::readCkpt(Finish copying restart data)");
}

void WriterPDI::writeXML(int step, Real time, bool plotMode) {
  std::string sd(plotMode ? "plot/" : "ckpt/");
  std::string filename = prefix + sd + filename_xmf;
  Print::Single(ps, "Writing file: {}\n", filename);

  // Create Xdmf file
  std::ofstream ofs(filename, std::ios::trunc);

  // Write header, open Xdmf and domain
  ofs << "<?xml version=\"1.0\" ?>\n"
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
      << "<Xdmf Version=\"2.0\">\n"
      << "  <Domain>" << std::endl;

  // Create grid, topology, and geometry
  ofs << "    <Grid Name=\"Structured Grid\" GridType=\"Uniform\">\n"
      << "      <Time Value=\"" << time << "\" />\n"
      << "      <Topology TopologyType=\"3DRectMesh\" NumberOfElements=\"" << (nZ_g + 1)
      << " " << (nY_g + 1) << " " << (nX_g + 1) << "\"/>\n"
      << "      <Geometry GeometryType=\"VxVyVz\">\n"
      << "        <DataItem Name=\"Vx\" Dimensions=\"" << (nX_g + 1)
      << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
      << "          " << filename_h5 << ":/xcoord\n"
      << "        </DataItem>\n"
      << "        <DataItem Name=\"Vy\" Dimensions=\"" << (nY_g + 1)
      << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
      << "          " << filename_h5 << ":/ycoord\n"
      << "        </DataItem>\n"
      << "        <DataItem Name=\"Vz\" Dimensions=\"" << (nZ_g + 1)
      << "\" NumberType=\"Float\" Precision=\"8\" Format=\"HDF\">\n"
      << "          " << filename_h5 << ":/zcoord\n"
      << "        </DataItem>\n      </Geometry>\n"
      << std::endl;

  // Write all data fields as attributes
  switch (eqType) {
  case EquationType::LinAdv:
    writeAttrLinAdv(ofs, plotMode);
    break;
  case EquationType::MHD_GLM:
    writeAttrMHD_GLM(ofs, plotMode);
    break;
  case EquationType::SRHydro:
    writeAttrSRHydro(ofs, plotMode);
    break;
  default:
    writeAttrHydro(ofs, plotMode);
  }
  writeAttributeScalar(ofs, "weno");
  writeAttributeScalar(ofs, "fomx");

  // Close grid, domain, Xdmf
  ofs << "    </Grid>\n  </Domain>\n</Xdmf>" << std::endl;

  // Close file
  ofs.close();
}

void WriterPDI::writeAttrHydro(std::ofstream &ofs, bool plotMode) {
  writeAttributeScalar(ofs, "dens");
  writeAttributeVector(ofs, "mom", "momx", "momy", "momz");
  writeAttributeScalar(ofs, "etot");
  if (plotMode) {
    writeAttributeVector(ofs, "vel", "velx", "vely", "velz");
    writeAttributeScalar(ofs, "eint");
    writeAttributeScalar(ofs, "pres");
  }
}

void WriterPDI::writeAttrLinAdv(std::ofstream &ofs, bool plotMode) {
  writeAttributeScalar(ofs, "dens");
}

void WriterPDI::writeAttrMHD_GLM(std::ofstream &ofs, bool plotMode) {
  writeAttributeScalar(ofs, "dens");
  writeAttributeVector(ofs, "mom", "momx", "momy", "momz");
  writeAttributeVector(ofs, "mag", "magx", "magy", "magz");
  writeAttributeScalar(ofs, "etot");
  writeAttributeScalar(ofs, "psi");
  if (plotMode) {
    writeAttributeVector(ofs, "vel", "velx", "vely", "velz");
    writeAttributeScalar(ofs, "eint");
    writeAttributeScalar(ofs, "prsg");
    writeAttributeScalar(ofs, "prsb");
  }
}

void WriterPDI::writeAttrSRHydro(std::ofstream &ofs, bool plotMode) {
  writeAttributeScalar(ofs, "dens");
  writeAttributeVector(ofs, "mom", "momx", "momy", "momz");
  writeAttributeScalar(ofs, "etot");
  if (plotMode) {
    writeAttributeScalar(ofs, "rho");
    writeAttributeVector(ofs, "vel", "velx", "vely", "velz");
    writeAttributeScalar(ofs, "lorz");
    writeAttributeScalar(ofs, "pres");
  }
}

void WriterPDI::writeAttributeScalar(std::ofstream &ofs, const char *varName) {
  ofs << "      <Attribute Name=\"" << varName
      << "\" AttributeType=\"Scalar\" Center=\"Cell\">\n"
      << "        <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" "
         "Dimensions=\""
      << nZ_g << " " << nY_g << " " << nX_g << "\">\n"
      << "          " << filename_h5 << ":/" << varName << "\n"
      << "        </DataItem>\n      </Attribute>\n"
      << std::endl;
}

void WriterPDI::writeAttributeVector(std::ofstream &ofs, const char *vecName,
                                     const char *vecX, const char *vecY,
                                     const char *vecZ) {
  ofs << "      <Attribute Name=\"" << vecName
      << "\" AttributeType=\"Vector\" Center=\"Cell\">\n"
      << "        <DataItem ItemType=\"Function\" Function=\"JOIN($0, $1, $2)\" "
         "Dimensions=\""
      << nZ_g << " " << nY_g << " " << nX_g
      << " 3\">\n"
      // x-component (permutes to the  component)
      << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" "
         "Dimensions=\""
      << nZ_g << " " << nY_g << " " << nX_g << "\">\n"
      << "            " << filename_h5 << ":/" << vecX << "\n"
      << "          </DataItem>\n"
      // y-component (permutes to the  component)
      << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" "
         "Dimensions=\""
      << nZ_g << " " << nY_g << " " << nX_g << "\">\n"
      << "            " << filename_h5 << ":/" << vecY << "\n"
      << "          </DataItem>\n"
      // z-component (permutes to the  component)
      << "          <DataItem Format=\"HDF\" NumberType=\"Float\" Precision=\"8\" "
         "Dimensions=\""
      << nZ_g << " " << nY_g << " " << nX_g << "\">\n"
      << "            " << filename_h5 << ":/" << vecZ << "\n"
      << "          </DataItem>\n"

      << "        </DataItem>\n      </Attribute>\n"
      << std::endl;
}

} // end namespace IO

} // end namespace KFVM
