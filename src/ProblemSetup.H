#ifndef PROBLEMSETUP_H_
#define PROBLEMSETUP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>

#include <fmt/color.h>
#include <fmt/core.h>

#include <mpi.h>

#include <Definitions.H>

#include "Dimension.H"
#include "LayoutMPI.H"
#include "ParameterStructs.H"
#include "Types.H"
#include "physics/EquationTypes.H"

namespace KFVM {

struct ProblemSetup {
  // Array of boundary conditions
  // Note: always 6 entries (2*3) even in 2D
  static std::array<BCType, 6> bcType;

  static bool haveSourceTerms;

  // MPI information
  LayoutMPI layoutMPI;

  // Stencil settings
  const idx_t rad;
  Real gp_lFac;

  // Number of blocks and size of each
  idx_t nbX, nbY, nbZ;
  idx_t nX, nY, nZ;

  // Global domain size
  Real xLo_g, xUp_g, yLo_g, yUp_g, zLo_g, zUp_g;

  // Time stepping information
  int maxTimeSteps;
  Real finalTime, minDeltaT;
  Real atol, rtol;
  Real rejectionThresh;
  int rejectionLimit;
  Real cfl;

  // File output information
  std::string pdiConf, dataDir, baseName;
  int ckptFreq, statsFreq;
  Real plotFreq;

  // Restart from checkpoint info
  bool restart;
  std::string restartFile;

  // Parameter structs for passing info to compute kernels
  EosParameters eosParams;
  UserParameters userParams;

  // Verbosity of output information
  int verbose;

  // Simple constructor setting many default values
  ProblemSetup()
      : rad(STENCIL_RADIUS), layoutMPI(), gp_lFac(20.0), nX(1), nY(1), nZ(1), nbX(1),
        nbY(1), nbZ(1), xLo_g(0.0), xUp_g(1.0), yLo_g(0.0), yUp_g(1.0), zLo_g(0.0),
        zUp_g(1.0), maxTimeSteps(1000), finalTime(1.0), minDeltaT(1.e-10), atol(1.e-4),
        rtol(1.e-4), rejectionThresh(0.8), rejectionLimit(4),
        cfl(SPACE_DIM == 3 ? 0.5 : 0.75), pdiConf(""), dataDir("data"),
        baseName("default"), plotFreq(-1.0), ckptFreq(-1), statsFreq(-1), restart(false),
        verbose(0) {
    // Ensure that source terms are always enabled in MHD
    haveSourceTerms = haveSourceTerms || (eqType == EquationType::MHD_GLM);
  }

  void print() {
    if (layoutMPI.rank == 0) {
      // Print mesh information
      fmt::print("Mesh:\n");
      fmt::print("  nX: {}\n", nX);
      fmt::print("  nY: {}\n", nY);
      fmt::print("  nZ: {}\n", nZ);
      fmt::print("  nbX: {}\n", nbX);
      fmt::print("  nbY: {}\n", nbY);
      fmt::print("  nbZ: {}\n", nbZ);
      fmt::print("  x extent: {} -- {}\n", xLo_g, xUp_g);
      fmt::print("  y extent: {} -- {}\n", yLo_g, yUp_g);
      fmt::print("  z extent: {} -- {}\n", zLo_g, zUp_g);

      // Print time information
      fmt::print("Time:\n");
      fmt::print("  final: {:<.4}\n", finalTime);
      fmt::print("  min dt: {:<.4}\n", minDeltaT);
      fmt::print("  atol: {:<.4}\n", atol);
      fmt::print("  rtol: {:<.4}\n", rtol);
      fmt::print("  rejectionlimit: {}\n", rejectionLimit);
      fmt::print("  rejectionthresh: {:<.4}\n", rejectionThresh);
      fmt::print("  cfl: {:f}\n", cfl);
      fmt::print("  maxsteps: {}\n", maxTimeSteps);

      // Print stencil information
      fmt::print("Stencil:\n");
      fmt::print("  Radius: {}\n", rad);
      fmt::print("  lfac: {:<.4}\n", gp_lFac);

      // Print eos parameters
      fmt::print("EOS:\n");
      fmt::print("  gamma: {:<.4}\n", eosParams.gamma);
      if (eqType == EquationType::NavierStokes) {
        fmt::print("  reynolds: {:f}\n", eosParams.Re);
        fmt::print("  prandtl: {:<.4}\n", eosParams.Pr);
      }
      fmt::print("  wenothresh: {:<.4}\n", eosParams.wenoThresh);
#ifdef ENABLE_RANDOM_FORCING
      fmt::print("  forceedot: {:<.4}\n", eosParams.forceEDot);
      fmt::print("  forcetoff: {:<.4}\n", eosParams.forceTOff);
      fmt::print("  forceratio: {:<.4}\n", eosParams.forceRatio);
#endif

      // Print io information
      fmt::print("IO:\n");
      fmt::print("  pdiconf: {}\n", pdiConf);
      fmt::print("  datadir: {}\n", dataDir);
      fmt::print("  basename: {}\n", baseName);
      fmt::print("  plotfreq: {}\n", plotFreq);
      fmt::print("  ckptfreq: {}\n", ckptFreq);
      fmt::print("  statsfreq: {}\n", statsFreq);
      if (restart) {
        fmt::print("  restartfile: {}\n", restartFile);
      }
    }
  }

  bool setFromFile(const char *fname) {
    // Open file and report success
    std::ifstream inFile;
    inFile.open(fname);
    bool opened = inFile.is_open();
    if (opened) {
      fmt::print("Setting options from input file: {}\n", fname);
    } else {
      fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                 "Error!!!! Could not open input file: {}\n", fname);
      return false;
    }

    // File opened successfully, start parsing through
    std::string line;
    while (std::getline(inFile, line)) {
      // Ignore comments and blank lines
      if (line[0] == '#' || line.empty()) {
        continue;
      }

      // Tokenize on :
      std::istringstream lStream(line);
      std::string category, setting;
      std::getline(lStream, category, ':');
      std::getline(lStream, setting, ':');
      bool success;

      // Chain if-elses to find category and forward the setting
      if (category == "mesh") {
        success = setMesh(setting);
      } else if (category == "stencil") {
        success = setStencil(setting);
      } else if (category == "time") {
        success = setTime(setting);
      } else if (category == "eos") {
        success = setEos(setting);
      } else if (category == "user") {
        success = setUser(setting);
      } else if (category == "io") {
        success = setIO(setting);
      } else {
        fmt::print(fmt::fg(fmt::color::golden_rod),
                   "[Warning] Ignoring line from input file: {}\n", line);
      }

      // Report malformed lines
      if (!success) {
        fmt::print(fmt::fg(fmt::color::golden_rod),
                   "[Warning] Line could not be parsed: {}\n", line);
      }
    }

    // setup the MPI layout, add better error handling later
    if (!layoutMPI.distributeBlocks(nbX, nbY, nbZ, nX, nY, nZ, bcType)) {
      fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red),
                 "[Error] Number of blocks must equal number of MPI ranks, have: "
                 "{}*{}*{} != {}\n",
                 nbX, nbY, nbZ, layoutMPI.size);
      return false;
    }

    return true;
  }

 private:
  bool setMesh(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;
    std::size_t pos;

    // category before space
    std::getline(lStream, category, ' ');

    if (category == "resolution") {
      std::getline(lStream, value, ' ');
      nX = std::stoi(value, &pos);
      std::getline(lStream, value, ' ');
      nY = std::stoi(value, &pos);
#if (SPACE_DIM == 3)
      std::getline(lStream, value, ' ');
      nZ = std::stoi(value, &pos);
#endif
    } else if (category == "numblocks") {
      std::getline(lStream, value, ' ');
      nbX = std::stoi(value, &pos);
      std::getline(lStream, value, ' ');
      nbY = std::stoi(value, &pos);
#if (SPACE_DIM == 3)
      std::getline(lStream, value, ' ');
      nbZ = std::stoi(value, &pos);
#endif
    } else if (category == "extent") {
      std::getline(lStream, value, ' ');
      xLo_g = std::stof(value, &pos);
      std::getline(lStream, value, ' ');
      xUp_g = std::stof(value, &pos);
      std::getline(lStream, value, ' ');
      yLo_g = std::stof(value, &pos);
      std::getline(lStream, value, ' ');
      yUp_g = std::stof(value, &pos);
#if (SPACE_DIM == 3)
      std::getline(lStream, value, ' ');
      zLo_g = std::stof(value, &pos);
      std::getline(lStream, value, ' ');
      zUp_g = std::stof(value, &pos);
#endif
    } else {
      return false;
    }

    return true;
  }

  bool setStencil(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;
    std::size_t pos;

    // category before space
    std::getline(lStream, category, ' ');

    if (category == "lfac") {
      std::getline(lStream, value, ' ');
      gp_lFac = std::stof(value, &pos);
    } else {
      return false;
    }

    return true;
  }

  bool setTime(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;
    std::size_t pos;

    // category before space
    std::getline(lStream, category, ' ');

    if (category == "mindt") {
      std::getline(lStream, value, ' ');
      minDeltaT = std::stof(value, &pos);
    } else if (category == "final") {
      std::getline(lStream, value, ' ');
      finalTime = std::stof(value, &pos);
    } else if (category == "cfl") {
      std::getline(lStream, value, ' ');
      cfl = std::stof(value, &pos);
    } else if (category == "atol") {
      std::getline(lStream, value, ' ');
      atol = std::stof(value, &pos);
    } else if (category == "rtol") {
      std::getline(lStream, value, ' ');
      rtol = std::stof(value, &pos);
    } else if (category == "rejectionthresh") {
      std::getline(lStream, value, ' ');
      rejectionThresh = std::stof(value, &pos);
    } else if (category == "rejectionlimit") {
      std::getline(lStream, value, ' ');
      rejectionLimit = std::stod(value, &pos);
    } else if (category == "maxsteps") {
      std::getline(lStream, value, ' ');
      maxTimeSteps = std::stod(value, &pos);
    } else {
      return false;
    }

    return true;
  }

  bool setEos(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;

    // Get category and value, forward to settings object
    std::getline(lStream, category, ' ');
    std::getline(lStream, value, ' ');
    return eosParams.set(category, value);
  }

  bool setUser(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;

    // Get category and value, forward to settings object
    std::getline(lStream, category, ' ');
    std::getline(lStream, value, ' ');
    return userParams.set(category, value);
  }

  bool setIO(const std::string &setting) {
    std::istringstream lStream(setting);
    std::string category, value;
    std::size_t pos;

    // category before space
    std::getline(lStream, category, ' ');

    if (category == "plotfreq") {
      std::getline(lStream, value, ' ');
      plotFreq = std::stof(value, &pos);
    } else if (category == "ckptfreq") {
      std::getline(lStream, value, ' ');
      ckptFreq = std::stoi(value, &pos);
    } else if (category == "statsfreq") {
      std::getline(lStream, value, ' ');
      statsFreq = std::stoi(value, &pos);
    } else if (category == "basename") {
      std::getline(lStream, baseName);
    } else if (category == "pdiconf") {
      std::getline(lStream, pdiConf);
    } else if (category == "datadir") {
      std::getline(lStream, dataDir);
    } else if (category == "restartfile") {
      std::getline(lStream, restartFile);
      restart = true;
    } else if (category == "verbose") {
      std::getline(lStream, value, ' ');
      verbose = std::stoi(value, &pos);
    } else {
      return false;
    }

    return true;
  }
};

} // namespace KFVM
#endif
