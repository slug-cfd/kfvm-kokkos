struct UserParameters : ParameterStructs {
  Real rhoH, rhoL, bFac;

  UserParameters() : rhoH(4.0), rhoL(1.0), bFac(0.2) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "rhoh") {
      rhoH = std::stof(val, &pos);
    } else if (fld == "rhol") {
      rhoL = std::stof(val, &pos);
    } else if (fld == "bfac") {
      bFac = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
