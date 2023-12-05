struct UserParameters : ParameterStructs {
  Real rhoH, rhoL, Rsq;

  UserParameters() : rhoH(4.0), rhoL(1.0), Rsq(0.1 * 0.1) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "rhoh") {
      rhoH = std::stof(val, &pos);
    } else if (fld == "rhol") {
      rhoL = std::stof(val, &pos);
    } else if (fld == "r") {
      Rsq = std::stof(val, &pos);
      Rsq *= Rsq;
    } else {
      return false;
    }

    return true;
  }
};
