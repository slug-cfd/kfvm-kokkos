struct UserParameters : ParameterStructs {
  int probType;

  UserParameters() : probType(0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "probtype") {
      probType = std::stod(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
