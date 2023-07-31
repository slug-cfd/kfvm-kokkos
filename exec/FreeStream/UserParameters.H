struct UserParameters : ParameterStructs {
  Real rpf, vx, vy, vz;

  UserParameters() : rpf(0.0), vx(0.0), vy(0.0), vz(0.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "rpf") {
      rpf = std::stof(val, &pos);
    } else if (fld == "vx") {
      vx = std::stof(val, &pos);
    } else if (fld == "vy") {
      vy = std::stof(val, &pos);
    } else if (fld == "vz") {
      vz = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
