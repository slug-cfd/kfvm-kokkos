struct UserParameters : ParameterStructs {
  Real A, beta;

  UserParameters() : beta(0.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "a") {
      A = std::stof(val, &pos);
    } else if (fld == "beta") {
      beta = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
