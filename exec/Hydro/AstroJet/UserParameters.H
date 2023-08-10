struct UserParameters : ParameterStructs {
  Real inletRadius, mach, densityRatio;

  UserParameters() : inletRadius(0.05), mach(800.0), densityRatio(10.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "inletradius") {
      inletRadius = std::stof(val, &pos);
    } else if (fld == "mach") {
      mach = std::stof(val, &pos);
    } else if (fld == "densityratio") {
      densityRatio = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
