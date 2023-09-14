struct UserParameters : ParameterStructs {
  Real mach, densRat;

  UserParameters() : mach(2.0), densRat(3.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "mach") {
      mach = std::stof(val, &pos);
    } else if (fld == "densrat") {
      densRat = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
