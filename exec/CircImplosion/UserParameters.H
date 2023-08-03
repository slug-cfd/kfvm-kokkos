struct UserParameters : ParameterStructs {
  Real A, M;

  UserParameters() : A(0.01), M(4.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "a") {
      A = std::stof(val, &pos);
    } else if (fld == "m") {
      M = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
