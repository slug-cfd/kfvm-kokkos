struct UserParameters : ParameterStructs {
  Real A0, R;

  UserParameters() : A0(1.e-3), R(0.3) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "a0") {
      A0 = std::stof(val, &pos);
    } else if (fld == "radius") {
      R = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
