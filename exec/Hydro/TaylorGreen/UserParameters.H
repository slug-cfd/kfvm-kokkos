struct UserParameters : ParameterStructs {
  Real mach;

  UserParameters() : mach(0.1) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "mach") {
      mach = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
