#ifndef SETTINGSCONTAINERS_H_
#define SETTINGSCONTAINERS_H_

// File: ParameterStructs.H
// Purpose: Hold all fluid properties in an easy to pass-by-value way

#include <cstddef>
#include <string>

#include <Definitions.H>

#include "Types.H"

namespace KFVM {

struct ParameterStructs {
  virtual bool set(const std::string &, const std::string &) = 0;
};

struct EosParameters : ParameterStructs {
  Real gamma;      // Ratio of specific heats
  Real wenoThresh; // Threshold for weno selector
  Real ch_glm;     // Hyperbolic cleaning speed for GLM

  EosParameters() : gamma(1.4), wenoThresh(10.0), ch_glm(0.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "gamma") {
      gamma = std::stof(val, &pos);
    } else if (fld == "wenothresh") {
      wenoThresh = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};

// Bring in user defined parameters
#include <UserParameters.H>

} // namespace KFVM

#endif
