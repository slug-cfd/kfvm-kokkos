struct UserParameters : ParameterStructs {
  Real R, pIn, pOut, bx;

  UserParameters() : R(0.0), pIn(0.0), pOut(0.0), bx(0.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "r") {
      R = std::stof(val, &pos);
    } else if (fld == "pin") {
      pIn = std::stof(val, &pos);
    } else if (fld == "pout") {
      pOut = std::stof(val, &pos);
    } else if (fld == "bx") {
      bx = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
