#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct UserParameters : ParameterStructs {
  Real ang, freq;

  UserParameters() : ang(0.0), freq(1.0) {}

  bool set(const std::string &fld, const std::string &val) {
    std::size_t pos;

    if (fld == "ang") {
      ang = M_PI * std::stof(val, &pos);
    } else if (fld == "freq") {
      freq = std::stof(val, &pos);
    } else {
      return false;
    }

    return true;
  }
};
