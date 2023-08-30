struct UserParameters : ParameterStructs {
  Real mach, k, delta;

  UserParameters() : mach(0.1), k(80.0), delta(0.05) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "mach") {
      mach = std::stof(val, &pos);
    } else if (fld == "k") {
      k = std::stof(val, &pos);
    } else if (fld == "delta") {
      delta = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
