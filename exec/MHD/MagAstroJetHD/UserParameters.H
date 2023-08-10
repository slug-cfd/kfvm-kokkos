struct UserParameters : ParameterStructs {
  Real inletRadius, mach, densityRatio, beta;

  UserParameters() : inletRadius(0.05), mach(800.0), densityRatio(10.0), beta(1.e-4) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "inletradius") {
      inletRadius = std::stof(val, &pos);
    } else if (fld == "mach") {
      mach = std::stof(val, &pos);
    } else if (fld == "densityratio") {
      densityRatio = std::stof(val, &pos);
    } else if (fld == "beta") {
      beta = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
