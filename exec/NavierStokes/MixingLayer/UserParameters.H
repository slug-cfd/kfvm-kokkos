struct UserParameters : ParameterStructs {
  Real omega;

  UserParameters() : omega(0.3147876) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "omega") {
      omega = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
