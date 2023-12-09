struct UserParameters : ParameterStructs {
  Real par;

  UserParameters(): par(0.0) {}

  bool set(const std::string& fld,const std::string& val)
  {
    std::size_t pos;
      
    if (fld == "par") {
      par = std::stof(val,&pos);
    } else {
      return false;
    }

    return true;
  }
};
