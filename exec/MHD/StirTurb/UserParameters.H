struct UserParameters : ParameterStructs {
  Real beta;

  UserParameters() : beta(1.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "beta") {
      beta = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
