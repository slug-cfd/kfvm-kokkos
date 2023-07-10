#include "StencilSymmetry.H"
#include "../Types.H"

namespace KFVM {

namespace Stencil {

// Conversion from symmetry type to name for printing
const char *symName(SymType st) {
  switch (st) {
  case SymType::RefX:
    return "RefX";
  case SymType::RefY:
    return "RefY";
  case SymType::RefZ:
    return "RefZ";
  case SymType::RefXY:
    return "RefXY";
  case SymType::RefXZ:
    return "RefXZ";
  case SymType::RefYZ:
    return "RefYZ";
  case SymType::RefXYZ:
    return "RefXYZ";
  case SymType::RotXY:
    return "RotXY";
  case SymType::RotXZ:
    return "RotXZ";
  case SymType::RotYZ:
    return "RotYZ";
  case SymType::RotXYm:
    return "RotXYm";
  case SymType::RotXZm:
    return "RotXZm";
  case SymType::RotYZm:
    return "RotYZm";
  default:
    return "None";
  }
}

// Functions to test for symmetry
template <SymType st>
bool testSymP(double, double, double, double);
template <SymType st>
bool testSymP(double, double, double, double, double, double);
bool dblEq(double a, double b) { return std::fabs(a - b) < 1.e-6; }

// Single reflections
template <>
bool testSymP<SymType::RefX>(double x1, double y1, double x2, double y2) {
  return dblEq(x1, -x2) && dblEq(y1, y2);
}

template <>
bool testSymP<SymType::RefX>(double x1, double y1, double z1, double x2, double y2,
                             double z2) {
  return dblEq(x1, -x2) && dblEq(y1, y2) && dblEq(z1, z2);
}

template <>
bool testSymP<SymType::RefY>(double x1, double y1, double x2, double y2) {
  return dblEq(x1, x2) && dblEq(y1, -y2);
}

template <>
bool testSymP<SymType::RefY>(double x1, double y1, double z1, double x2, double y2,
                             double z2) {
  return dblEq(x1, x2) && dblEq(y1, -y2) && dblEq(z1, z2);
}

template <>
bool testSymP<SymType::RefZ>(double x1, double y1, double z1, double x2, double y2,
                             double z2) {
  return dblEq(x1, x2) && dblEq(y1, y2) && dblEq(z1, -z2);
}

// Double reflections
template <>
bool testSymP<SymType::RefXY>(double x1, double y1, double x2, double y2) {
  return dblEq(x1, -x2) && dblEq(y1, -y2);
}

template <>
bool testSymP<SymType::RefXY>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, -x2) && dblEq(y1, -y2) && dblEq(z1, z2);
}

template <>
bool testSymP<SymType::RefXZ>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, -x2) && dblEq(y1, y2) && dblEq(z1, -z2);
}

template <>
bool testSymP<SymType::RefYZ>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, x2) && dblEq(y1, -y2) && dblEq(z1, -z2);
}

// Triple reflection
template <>
bool testSymP<SymType::RefXYZ>(double x1, double y1, double z1, double x2, double y2,
                               double z2) {
  return dblEq(x1, -x2) && dblEq(y1, -y2) && dblEq(z1, -z2);
}

// Planar rotations
template <>
bool testSymP<SymType::RotXY>(double x1, double y1, double x2, double y2) {
  return dblEq(x1, -y2) && dblEq(y1, x2);
}

template <>
bool testSymP<SymType::RotXYm>(double x1, double y1, double x2, double y2) {
  return dblEq(x1, y2) && dblEq(y1, -x2);
}

template <>
bool testSymP<SymType::RotXY>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, -y2) && dblEq(y1, x2) && dblEq(z1, z2);
}

template <>
bool testSymP<SymType::RotXYm>(double x1, double y1, double z1, double x2, double y2,
                               double z2) {
  return dblEq(x1, y2) && dblEq(y1, -x2) && dblEq(z1, z2);
}

template <>
bool testSymP<SymType::RotXZ>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, -z2) && dblEq(y1, y2) && dblEq(z1, x2);
}

template <>
bool testSymP<SymType::RotXZm>(double x1, double y1, double z1, double x2, double y2,
                               double z2) {
  return dblEq(x1, z2) && dblEq(y1, y2) && dblEq(z1, -x2);
}

template <>
bool testSymP<SymType::RotYZ>(double x1, double y1, double z1, double x2, double y2,
                              double z2) {
  return dblEq(x1, x2) && dblEq(y1, -z2) && dblEq(z1, y2);
}

template <>
bool testSymP<SymType::RotYZm>(double x1, double y1, double z1, double x2, double y2,
                               double z2) {
  return dblEq(x1, x2) && dblEq(y1, z2) && dblEq(z1, -y2);
}

// Aggregate test
bool testSym(SymType st, double x1, double y1, double x2, double y2) {
  switch (st) {
  case SymType::RefX:
    return testSymP<SymType::RefX>(x1, y1, x2, y2);
  case SymType::RefY:
    return testSymP<SymType::RefY>(x1, y1, x2, y2);
  case SymType::RefXY:
    return testSymP<SymType::RefXY>(x1, y1, x2, y2);
  case SymType::RotXY:
    return testSymP<SymType::RotXY>(x1, y1, x2, y2);
  case SymType::RotXYm:
    return testSymP<SymType::RotXYm>(x1, y1, x2, y2);
  default:
    return false;
  }
}

bool testSym(SymType st, double x1, double y1, double z1, double x2, double y2,
             double z2) {
  switch (st) {
  case SymType::RefX:
    return testSymP<SymType::RefX>(x1, y1, z1, x2, y2, z2);
  case SymType::RefY:
    return testSymP<SymType::RefY>(x1, y1, z1, x2, y2, z2);
  case SymType::RefZ:
    return testSymP<SymType::RefZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RefXY:
    return testSymP<SymType::RefXY>(x1, y1, z1, x2, y2, z2);
  case SymType::RefXZ:
    return testSymP<SymType::RefXZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RefYZ:
    return testSymP<SymType::RefYZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RefXYZ:
    return testSymP<SymType::RefXYZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RotXY:
    return testSymP<SymType::RotXY>(x1, y1, z1, x2, y2, z2);
  case SymType::RotXZ:
    return testSymP<SymType::RotXZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RotYZ:
    return testSymP<SymType::RotYZ>(x1, y1, z1, x2, y2, z2);
  case SymType::RotXYm:
    return testSymP<SymType::RotXYm>(x1, y1, z1, x2, y2, z2);
  case SymType::RotXZm:
    return testSymP<SymType::RotXZm>(x1, y1, z1, x2, y2, z2);
  case SymType::RotYZm:
    return testSymP<SymType::RotYZm>(x1, y1, z1, x2, y2, z2);
  default:
    return false;
  }
}

// Symmetry between two stencils
StencilSymmetry::StencilSymmetry(
    bool isSelf,
    KFVM_D_DECL(const std::vector<double> &lOffD, const std::vector<double> &tOffD,
                const std::vector<double> &ttOffD),
    KFVM_D_DECL(const std::vector<double> &lOffR, const std::vector<double> &tOffR,
                const std::vector<double> &ttOffR),
    KFVM_D_DECL(const std::vector<double> &xD, const std::vector<double> &yD,
                const std::vector<double> &zD),
    KFVM_D_DECL(const std::vector<double> &xR, const std::vector<double> &yR,
                const std::vector<double> &zR))
    : nPts(xD.size()), N(lOffD.size()), symMap(nPts, SymMap(-1, SymType::None)),
      idxMap(nPts, std::vector<idx_t>(N, -1)) {
  // Find all symmetries that the stencil(s) have
  for (unsigned int i = 0; i < SymTypes.size(); i++) {
    // Loop over all destination coordinates and check symmetry i
    bool hasSym = true;
    for (idx_t nD = 0; nD < N; nD++) {
      bool dSym = false;
      // Find reference point
      for (idx_t nR = 0; nR < N; nR++) {
        dSym = testSym(SymTypes[i], KFVM_D_DECL(lOffD[nD], tOffD[nD], ttOffD[nD]),
                       KFVM_D_DECL(lOffR[nR], tOffR[nR], ttOffR[nR]));
        if (dSym) {
          break;
        }
      }
      hasSym = hasSym && dSym;
    }
    if (hasSym) {
      stenSym.push_back(SymTypes[i]);
    }
  }

  // Find all symmetries the query points have
  // but restrict them to symmetries the underlying stencil has
  for (idx_t nD = 0; nD < nPts; nD++) {
    // Try to map point nD onto point nR
    idx_t refLim = isSelf ? nD : nPts;
    for (idx_t nR = 0; nR < refLim; nR++) {
      // and test only allowed symmetries
      bool hasSym = false;
      for (unsigned int i = 0; i < stenSym.size(); i++) {
        hasSym = testSym(stenSym[i], KFVM_D_DECL(xD[nD], yD[nD], zD[nD]),
                         KFVM_D_DECL(xR[nR], yR[nR], zR[nR]));
        if (hasSym) {
          symMap[nD].first = nR;
          symMap[nD].second = stenSym[i];
          break;
        }
      }
      // Break and process next destination point
      if (hasSym) {
        break;
      }
    }
  }

  // Finally, form index maps over the stencils
  for (idx_t nP = 0; nP < nPts; nP++) {
    // Skip if this point isn't given by another
    if (symMap[nP].first < 0) {
      continue;
    }

    // Otherwise find coord nR matching coord nD
    for (idx_t nD = 0; nD < N; nD++) {
      for (idx_t nR = 0; nR < N; nR++) {
        if (testSym(symMap[nP].second, KFVM_D_DECL(lOffD[nD], tOffD[nD], ttOffD[nD]),
                    KFVM_D_DECL(lOffR[nR], tOffR[nR], ttOffR[nR]))) {
          idxMap[nP][nD] = nR;
          break;
        }
      }
    }
  }
}

} // end namespace Stencil

} // end namespace KFVM
