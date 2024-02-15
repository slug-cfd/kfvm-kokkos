struct UserParameters : ParameterStructs {
  Real bn, bt;

  UserParameters() : bn(0.75), bt(1.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "bn") {
      bn = std::stof(val, &pos);
    } else if (fld == "bt") {
      bt = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
