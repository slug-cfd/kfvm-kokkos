#ifndef PROBLEMSETUP_H_
#define PROBLEMSETUP_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#include <Definitions.H>

#include "Dimension.H"
#include "Types.H"
#include "physics/EquationTypes.H"
#include "FluidProperties.H"

#ifndef STENCIL_RADIUS
#define STENCIL_RADIUS 2
#warning Defaulted to radius 2 stencil
#endif

namespace KFVM {

  struct ProblemSetup
  {
    // Static members to be set by the user
    // Names for variables used in the solution files
    static std::array<std::string,NUM_VARS> varName;
    static std::array<std::string,NUM_AUX> auxVarName;

    // Array of boundary conditions
    // Note: always 6 entries (2*3) even in 2D
    static std::array<BCType,6> bcType;
    
    // Stencil settings
    const idx_t rad;
    Real gp_lFac,gp_bpow;
    
    // Domain and grid size
    idx_t nX,nY,nZ;
    Real xLo,xUp,yLo,yUp,zLo,zUp;

    // Time stepping information
    int maxTimeSteps;
    Real initialTime,finalTime,initialDeltaT;
    Real atol,rtol;
    Real rejectionThresh;
    int rejectionLimit;
    Real cfl;

    // File output information
    std::string pdiConf,dataDir,baseName;
    int plotFreq;

    // Fluid properties object
    FluidProperties fluidProp;

    // Simple constructor setting many default values
    ProblemSetup():
      rad(STENCIL_RADIUS),
      gp_lFac(20.0),
      gp_bpow(Real(STENCIL_RADIUS)),
      nX(1),
      nY(1),
      nZ(1),
      xLo(0.0),
      xUp(1.0),
      yLo(0.0),
      yUp(1.0),
      zLo(0.0),
      zUp(1.0),
      maxTimeSteps(1000),
      initialTime(0.0),
      finalTime(1.0),
      initialDeltaT(1.e-10),
      atol(1.e-4),
      rtol(1.e-4),
      rejectionThresh(0.8),
      rejectionLimit(4),
      cfl(SPACE_DIM==3 ? 0.5 : 0.75),
      pdiConf(""),
      dataDir("data"),
      baseName("default"),
      plotFreq(100)
    {}

    
    void print()
    {
      // Print mesh information
      std::printf("Mesh:\n");
      std::printf("  nX: %ld\n",nX);
      std::printf("  nY: %ld\n",nY);
      std::printf("  nZ: %ld\n",nZ);
      std::printf("  x extent: %f -- %f\n",xLo,xUp);
      std::printf("  y extent: %f -- %f\n",yLo,yUp);
      std::printf("  z extent: %f -- %f\n",zLo,zUp);
      
      // Print time information
      std::printf("Time:\n");
      std::printf("  initial: %f\n",initialTime);
      std::printf("  final: %f\n",finalTime);
      std::printf("  initial dt: %e\n",initialDeltaT);
      std::printf("  atol: %e\n",atol);
      std::printf("  rtol: %e\n",rtol);
      std::printf("  rejectionlimit: %d\n",rejectionLimit);
      std::printf("  rejectionthresh: %f\n",rejectionThresh);
      std::printf("  cfl: %f\n",cfl);
      std::printf("  maxsteps: %d\n",maxTimeSteps);

      // Print stencil information
      std::printf("Stencil:\n");
      std::printf("  Radius: %ld\n",rad);
      std::printf("  lfac: %f\n",gp_lFac);
      std::printf("  bpow: %f\n",gp_bpow);

      // Print fluid properties
      std::printf("Properties:\n");
      std::printf("  gamma: %f\n",fluidProp.gamma);
      std::printf("  wenothresh: %f\n",fluidProp.wenoThresh);

      // Print output information
      std::printf("Output:\n");
      std::printf("  pdiconf: %s\n",pdiConf.c_str());
      std::printf("  datadir: %s\n",dataDir.c_str());
      std::printf("  basename: %s\n",baseName.c_str());
      std::printf("  plotfreq: %d\n",plotFreq);
    }

    bool setFromFile(const char* fname)
    {
      // Open file and report success
      std::ifstream inFile;
      inFile.open(fname);
      bool opened = inFile.is_open();
      if (opened) {
	std::printf("Setting options from input file: %s\n",fname);
      } else {
	std::printf("Warning!!!! Could not open input file: %s\n",fname);
	return false;
      }

      // File opened successfully, start parsing through
      std::string line;
      while (std::getline(inFile,line)) {
	// Ignore comments and blank lines
	if (line[0] == '#' || line.empty()) {
	  continue;
	}

	// Tokenize on :
	std::istringstream lStream(line);
	std::string category,setting;
	std::getline(lStream,category,':');
	std::getline(lStream,setting,':');
	bool success;

	// Chain if-elses to find category and forward the setting
	if (category == "mesh") {
	  success = setMesh(setting);
	} else if (category == "stencil") {
	  success = setStencil(setting);
	} else if (category == "time") {
	  success = setTime(setting);
	} else if (category == "properties") {
	  success = setProperties(setting);
	} else if (category == "output") {
	  success = setOutput(setting);
	} else {
	  std::cout << "Warning!!!! Ignoring line from input file: " << line << std::endl;
	}

	// Report malformed lines
	if (!success) {
	  std::cout << "Warning!!!! Line could not be parsed: " << line << std::endl;
	}
      }
      return true;
    }

  private:
    
    bool setMesh(const std::string& setting) {
      std::istringstream lStream(setting);
      std::string category,value;
      std::size_t pos;
      
      // category before space
      std::getline(lStream,category,' ');

      if (category == "resolution") {
	std::getline(lStream,value,' ');
	nX = std::stoi(value,&pos);
	std::getline(lStream,value,' ');
	nY = std::stoi(value,&pos);
#if (SPACE_DIM == 3)
	std::getline(lStream,value,' ');
	nZ = std::stoi(value,&pos);
#endif
      } else if (category == "extent") {
	std::getline(lStream,value,' ');
	xLo = std::stof(value,&pos);
	std::getline(lStream,value,' ');
	xUp = std::stof(value,&pos);
	std::getline(lStream,value,' ');
	yLo = std::stof(value,&pos);
	std::getline(lStream,value,' ');
	yUp = std::stof(value,&pos);
#if (SPACE_DIM == 3)
	std::getline(lStream,value,' ');
	zLo = std::stof(value,&pos);
	std::getline(lStream,value,' ');
	zUp = std::stof(value,&pos);
#endif
      } else {
	return false;
      }

      return true;
    }
    
    bool setStencil(const std::string& setting) {
      std::istringstream lStream(setting);
      std::string category,value;
      std::size_t pos;
      
      // category before space
      std::getline(lStream,category,' ');

      if (category == "lfac") {
	std::getline(lStream,value,' ');
	gp_lFac = std::stof(value,&pos);
      } else if (category == "bpow") {
	std::getline(lStream,value,' ');
	gp_bpow = std::stof(value,&pos);
      } else {
	return false;
      }
      
      return true;
    }
    
    bool setTime(const std::string& setting) {
      std::istringstream lStream(setting);
      std::string category,value;
      std::size_t pos;
      
      // category before space
      std::getline(lStream,category,' ');

      if (category == "initial") {
	std::getline(lStream,value,' ');
	initialTime = std::stof(value,&pos);
      } else if (category == "initialdt") {
	std::getline(lStream,value,' ');
	initialDeltaT = std::stof(value,&pos);
      } else if (category == "final") {
	std::getline(lStream,value,' ');
	finalTime = std::stof(value,&pos);
      } else if (category == "cfl") {
	std::getline(lStream,value,' ');
	cfl = std::stof(value,&pos);
      } else if (category == "atol") {
	std::getline(lStream,value,' ');
	atol = std::stof(value,&pos);
      } else if (category == "rtol") {
	std::getline(lStream,value,' ');
	rtol = std::stof(value,&pos);
      } else if (category == "rejectionthresh") {
	std::getline(lStream,value,' ');
	rejectionThresh = std::stof(value,&pos);
      } else if (category == "rejectionlimit") {
	std::getline(lStream,value,' ');
	rejectionLimit = std::stod(value,&pos);
      } else if (category == "maxsteps") {
	std::getline(lStream,value,' ');
	maxTimeSteps = std::stod(value,&pos);
      } else {
	return false;
      }

      return true;
    }
    
    bool setProperties(const std::string& setting) {
      std::istringstream lStream(setting);
      std::string category,value;
      std::size_t pos;
      
      // category before space
      std::getline(lStream,category,' ');

      if (category == "gamma") {
	std::getline(lStream,value,' ');
	fluidProp.gamma = std::stof(value,&pos);
      } else if (category == "wenothresh") {
	std::getline(lStream,value,' ');
	fluidProp.wenoThresh = std::stof(value,&pos);
      } else {
	return false;
      }
      
      return true;
    }
    
    bool setOutput(const std::string& setting) {
      std::istringstream lStream(setting);
      std::string category,value;
      std::size_t pos;
      
      // category before space
      std::getline(lStream,category,' ');

      if (category == "plotfreq") {
	std::getline(lStream,value,' ');
	plotFreq = std::stoi(value,&pos);
      } else if (category == "basename") {
	std::getline(lStream,baseName);
      } else if (category == "pdiconf") {
	std::getline(lStream,pdiConf);
      } else if (category == "datadir") {
	std::getline(lStream,dataDir);
      } else {
	return false;
      }
      
      return true;
    }
    
  };
  
}
#endif
