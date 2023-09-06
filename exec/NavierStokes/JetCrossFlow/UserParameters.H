struct UserParameters : ParameterStructs {
  Real machFree, machJet, densRat, presRat;

  UserParameters() : machFree(0.8), machJet(3.73), densRat(1.0), presRat(1.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "machfree") {
      machFree = std::stof(val, &pos);
    } else if (fld == "machjet") {
      machJet = std::stof(val, &pos);
    } else if (fld == "densrat") {
      densRat = std::stof(val, &pos);
    } else if (fld == "presrat") {
      presRat = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
