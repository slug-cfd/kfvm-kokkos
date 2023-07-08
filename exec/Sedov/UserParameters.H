struct UserParameters : ParameterStructs
{
  Real R; // Sedov pulse radius

  UserParameters(): R(4.0/256.0) {}

  bool set(const std::string& fld,const std::string& val)
  {
    std::size_t pos;
      
    if (fld == "r") {
      R = std::stof(val,&pos);
    } else {
      return false;
    }

    return true;
  }
};
