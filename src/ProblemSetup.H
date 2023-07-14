#ifndef PROBLEMSETUP_H_
#define PROBLEMSETUP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>

#include <mpi.h>

#include <Definitions.H>

#include "Dimension.H"
#include "GeometryTypes.H"
#include "LayoutMPI.H"
#include "ParameterStructs.H"
#include "Types.H"
#include "physics/EquationTypes.H"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  Real gp_lFac, gp_bpow;

  // Number of blocks and size of each
  idx_t nbX, nbY, nbZ;
  idx_t nX, nY, nZ;

  // Global domain size
  Real xLo_g, xUp_g, yLo_g, yUp_g, zLo_g, zUp_g;

  // Time stepping information
  int maxTimeSteps;
  Real initialTime, finalTime, initialDeltaT;
  Real atol, rtol;
  Real rejectionThresh;
  int rejectionLimit;
  Real cfl;

  // File output information
  std::string pdiConf, dataDir, baseName;
  int plotFreq, ckptFreq;

  // Restart from checkpoint info
  bool restart;
  std::string restartFile;

  // Parameter structs for passing info to compute kernels
  EosParameters eosParams;
  UserParameters userParams;

  // Simple constructor setting many default values
  ProblemSetup()
      : rad(STENCIL_RADIUS), layoutMPI(), gp_lFac(20.0), gp_bpow(Real(STENCIL_RADIUS)),
        nX(1), nY(1), nZ(1), nbX(1), nbY(1), nbZ(1), xLo_g(0.0), xUp_g(1.0), yLo_g(0.0),
        yUp_g(1.0), zLo_g(0.0), zUp_g(1.0), maxTimeSteps(1000), initialTime(0.0),
        finalTime(1.0), initialDeltaT(1.e-10), atol(1.e-4), rtol(1.e-4),
        rejectionThresh(0.8), rejectionLimit(4), cfl(SPACE_DIM == 3 ? 0.5 : 0.75),
        pdiConf(""), dataDir("data"), baseName("default"), plotFreq(100), ckptFreq(50),
        restart(false) {
    // Ensure that source terms are always enabled in MHD
    haveSourceTerms = haveSourceTerms || (eqType == EquationType::MHD_GLM);
  }

  void print() {
    if (layoutMPI.rank == 0) {
      // Print mesh information
      std::printf("Mesh:\n");
      std::printf("  nX: %ld\n", nX);
      std::printf("  nY: %ld\n", nY);
      std::printf("  nZ: %ld\n", nZ);
      std::printf("  nbX: %ld\n", nbX);
      std::printf("  nbY: %ld\n", nbY);
      std::printf("  nbZ: %ld\n", nbZ);
      std::printf("  x extent: %f -- %f\n", xLo_g, xUp_g);
      std::printf("  y extent: %f -- %f\n", yLo_g, yUp_g);
      std::printf("  z extent: %f -- %f\n", zLo_g, zUp_g);

      // Print time information
      std::printf("Time:\n");
      std::printf("  initial: %f\n", initialTime);
      std::printf("  final: %f\n", finalTime);
      std::printf("  initial dt: %e\n", initialDeltaT);
      std::printf("  atol: %e\n", atol);
      std::printf("  rtol: %e\n", rtol);
      std::printf("  rejectionlimit: %d\n", rejectionLimit);
      std::printf("  rejectionthresh: %f\n", rejectionThresh);
      std::printf("  cfl: %f\n", cfl);
      std::printf("  maxsteps: %d\n", maxTimeSteps);

      // Print stencil information
      std::printf("Stencil:\n");
      std::printf("  Radius: %ld\n", rad);
      std::printf("  lfac: %f\n", gp_lFac);
      std::printf("  bpow: %f\n", gp_bpow);

      // Print eos parameters
      std::printf("EOS:\n");
      std::printf("  gamma: %f\n", eosParams.gamma);
      std::printf("  wenothresh: %f\n", eosParams.wenoThresh);

      // Print io information
      std::printf("IO:\n");
      std::printf("  pdiconf: %s\n", pdiConf.c_str());
      std::printf("  datadir: %s\n", dataDir.c_str());
      std::printf("  basename: %s\n", baseName.c_str());
      std::printf("  plotfreq: %d\n", plotFreq);
      std::printf("  ckptfreq: %d\n", ckptFreq);
      if (restart) {
        std::printf("  restartfile: %s\n", restartFile.c_str());
      }
    }
  }

  bool setFromFile(const char *fname) {
    // Open file and report success
    std::ifstream inFile;
    inFile.open(fname);
    bool opened = inFile.is_open();
    if (opened) {
      std::printf("Setting options from input file: %s\n", fname);
    } else {
      std::printf("Error!!!! Could not open input file: %s\n", fname);
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
        std::cout << "Warning!!!! Ignoring line from input file: " << line << std::endl;
      }

      // Report malformed lines
      if (!success) {
        std::cout << "Warning!!!! Line could not be parsed: " << line << std::endl;
      }
    }

    // Convert bounds to pi multiples if appropriate
    if (geomType == GeometryType::Polar) {
      yLo_g *= M_PI;
      yUp_g *= M_PI;
    }

    // setup the MPI layout, add better error handling later
    if (!layoutMPI.distributeBlocks(nbX, nbY, nbZ, nX, nY, nZ, bcType)) {
      std::printf("Error!!!! Number of blocks must equal number of MPI ranks, have: "
                  "%ld*%ld*%ld != %d\n",
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
    } else if (category == "bpow") {
      std::getline(lStream, value, ' ');
      gp_bpow = std::stof(value, &pos);
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

    if (category == "initial") {
      std::getline(lStream, value, ' ');
      initialTime = std::stof(value, &pos);
    } else if (category == "initialdt") {
      std::getline(lStream, value, ' ');
      initialDeltaT = std::stof(value, &pos);
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
      plotFreq = std::stoi(value, &pos);
    } else if (category == "ckptfreq") {
      std::getline(lStream, value, ' ');
      ckptFreq = std::stoi(value, &pos);
    } else if (category == "basename") {
      std::getline(lStream, baseName);
    } else if (category == "pdiconf") {
      std::getline(lStream, pdiConf);
    } else if (category == "datadir") {
      std::getline(lStream, dataDir);
    } else if (category == "restartfile") {
      std::getline(lStream, restartFile);
      restart = true;
    } else {
      return false;
    }

    return true;
  }
};

} // namespace KFVM
#endif
