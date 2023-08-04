#ifndef STENCILSYMMETRY_H_
#define STENCILSYMMETRY_H_

#include <array>
#include <utility>
#include <vector>

#include <fmt/color.h>
#include <fmt/core.h>

#include "../Types.H"

namespace KFVM {

namespace Stencil {

// The needed symmetries are just combinations of reflections and rotations
enum class SymType {
  RefX,
  RefY,
  RefZ,
  RefXY,
  RefXZ,
  RefYZ,
  RefXYZ,
  RotXY,
  RotXZ,
  RotYZ,
  RotXYm,
  RotXZm,
  RotYZm,
  None
};

// This array is needed because iterating on enums is impossible,
// but an enum is nice for this purpose otherwise
constexpr std::array<SymType, 14> SymTypes = {
    SymType::RefX,   SymType::RefY,   SymType::RefZ,   SymType::RefXY, SymType::RefXZ,
    SymType::RefYZ,  SymType::RefXYZ, SymType::RotXY,  SymType::RotXZ, SymType::RotYZ,
    SymType::RotXYm, SymType::RotXZm, SymType::RotYZm, SymType::None};

// Convert symmetry type to string for printing
const char *symName(SymType);

// Aggregate test
bool testSym(SymType st, double x1, double x2, double y1, double y2);
bool testSym(SymType st, double x1, double x2, double y1, double y2, double z1,
             double z2);

// Pair of reference point, and what symmetry operation gets there
typedef std::pair<idx_t, SymType> SymMap;

struct StencilSymmetry {
  // Index maps for each point according to its symmetry relation
  idx_t nPts, N;
  std::vector<SymType> stenSym;
  std::vector<SymMap> symMap;
  std::vector<std::vector<idx_t>> idxMap;

  StencilSymmetry() = delete;

  StencilSymmetry(bool isSelf,
                  KFVM_D_DECL(const std::vector<double> &, const std::vector<double> &,
                              const std::vector<double> &),
                  KFVM_D_DECL(const std::vector<double> &, const std::vector<double> &,
                              const std::vector<double> &),
                  KFVM_D_DECL(const std::vector<double> &, const std::vector<double> &,
                              const std::vector<double> &),
                  KFVM_D_DECL(const std::vector<double> &, const std::vector<double> &,
                              const std::vector<double> &));

  // Map one set of weights onto another
  template <class DstView, class RefView>
  void mapWeights(DstView &dWts, const RefView &rWts) {
    for (int nP = 0; nP < nPts; nP++) {
      if (symMap[nP].first >= 0) {
        // Fill destination from reference
        for (int j = 0; j < N; j++) {
          dWts(nP, j) = rWts(symMap[nP].first, idxMap[nP][j]);
        }
      } else {
        fmt::print(fmt::fg(fmt::color::golden_rod),
                   "Symmetry missing for {} in StencilSymmetry::mapWeights\n", nP);
      }
    }
  }
};

} // end namespace Stencil

} // end namespace KFVM

#endif
