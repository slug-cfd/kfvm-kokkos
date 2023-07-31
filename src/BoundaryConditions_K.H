#ifndef BOUNDARYCONDITIONS_K_H_
#define BOUNDARYCONDITIONS_K_H_

// File: BoundaryConditions_K.H
// Purpose: Define various boundary conditions for cell and face data
#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "Geometry.H"
#include "ParameterStructs.H"
#include "Types.H"
#include "numeric/Numeric.H"
#include "physics/EquationTypes.H"
#include "physics/Physics_K.H"

namespace KFVM {

namespace BoundaryConditions {

// Unspecialized templates
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcWest_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcEast_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcSouth_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcNorth_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcBottom_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct CellBcTop_K {};

template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcWest_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcEast_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcSouth_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcNorth_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcBottom_K {};
template <class DataViewType, BCType bcType, class RefCoeff = void>
struct FaceBcTop_K {};

// Bring in user defined BCs
namespace {
#include "UserBCs.H"
}

// Cell outflow conditions
template <class DataViewType>
struct CellBcWest_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nX;

  CellBcWest_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
               idx_t nX_)
      : v(v_), geom(geom_), rad(rad_), nX(nX_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(rad, j, k), nV) / geom.jacobian(KFVM_D_DECL(0, j, k));
      itf[nV] -=
          0.5 * v(KFVM_D_DECL(rad + 1, j, k), nV) / geom.jacobian(KFVM_D_DECL(1, j, k));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(nH, j, k));
      const Real dJ = geom.jacobian(KFVM_D_DECL(-nH - 1, j, k));
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(rad - nH - 1, j, k), nV) =
            dJ * (2.0 * itf[nV] - v(KFVM_D_DECL(rad + nH, j, k), nV) / sJ);
      }
    }
  }
};

template <class DataViewType>
struct CellBcEast_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nX;

  CellBcEast_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
               idx_t nX_)
      : v(v_), geom(geom_), rad(rad_), nX(nX_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(nX + rad - 1, j, k), nV) /
                geom.jacobian(KFVM_D_DECL(nX - 1, j, k));
      itf[nV] -= 0.5 * v(KFVM_D_DECL(nX + rad - 2, j, k), nV) /
                 geom.jacobian(KFVM_D_DECL(nX - 2, j, k));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(nX - nH - 1, j, k));
      const Real dJ = geom.jacobian(KFVM_D_DECL(nX + nH, j, k));
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(nX + rad + nH, j, k), nV) =
            dJ * (2.0 * itf[nV] - v(KFVM_D_DECL(nX + rad - nH - 1, j, k), nV) / sJ);
      }
    }
  }
};

template <class DataViewType>
struct CellBcSouth_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nY;

  CellBcSouth_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                idx_t nY_)
      : v(v_), geom(geom_), rad(rad_), nY(nY_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(i, rad, k), nV) / geom.jacobian(KFVM_D_DECL(i, 0, k));
      itf[nV] -=
          0.5 * v(KFVM_D_DECL(i, rad + 1, k), nV) / geom.jacobian(KFVM_D_DECL(i, 1, k));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, nH, k));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, -nH - 1, k));
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, rad - nH - 1, k), nV) =
            dJ * (2.0 * itf[nV] - v(KFVM_D_DECL(i, rad + nH, k), nV) / sJ);
      }
    }
  }
};

template <class DataViewType>
struct CellBcNorth_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nY;

  CellBcNorth_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                idx_t nY_)
      : v(v_), geom(geom_), rad(rad_), nY(nY_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(i, nY + rad - 1, k), nV) /
                geom.jacobian(KFVM_D_DECL(i, nY - 1, k));
      itf[nV] -= 0.5 * v(KFVM_D_DECL(i, nY + rad - 2, k), nV) /
                 geom.jacobian(KFVM_D_DECL(i, nY - 2, k));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, nY - nH - 1, k));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, nY + nH, k));
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, nY + rad + nH, k), nV) =
            dJ * (2.0 * itf[nV] - v(KFVM_D_DECL(i, nY + rad - nH - 1, k), nV) / sJ);
      }
    }
  }
};

template <class DataViewType>
struct CellBcBottom_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nZ;

  CellBcBottom_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                 idx_t nZ_)
      : v(v_), geom(geom_), rad(rad_), nZ(nZ_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(i, j, rad), nV) / geom.jacobian(KFVM_D_DECL(i, j, 0));
      itf[nV] -=
          0.5 * v(KFVM_D_DECL(i, j, rad + 1), nV) / geom.jacobian(KFVM_D_DECL(i, j, 1));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(i, j, nH);
      const Real dJ = geom.jacobian(i, j, -nH - 1);
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, rad - nH - 1, nV) = dJ * (2.0 * itf[nV] - v(i, j, rad + nH, nV) / sJ);
      }
    }
  }
};

template <class DataViewType>
struct CellBcTop_K<DataViewType, BCType::outflow> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nZ;

  CellBcTop_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
              idx_t nZ_)
      : v(v_), geom(geom_), rad(rad_), nZ(nZ_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Interface value to expand around
    Real itf[NUM_VARS];
    for (int nV = 0; nV < NUM_VARS; nV++) {
      itf[nV] = 1.5 * v(KFVM_D_DECL(i, j, nZ + rad - 1), nV) /
                geom.jacobian(KFVM_D_DECL(i, j, nZ - 1));
      itf[nV] -= 0.5 * v(KFVM_D_DECL(i, j, nZ + rad - 2), nV) /
                 geom.jacobian(KFVM_D_DECL(i, j, nZ - 2));
    }

    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(i, j, nZ - nH - 1);
      const Real dJ = geom.jacobian(i, j, nZ + nH);
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, nZ + rad + nH, nV) =
            dJ * (2.0 * itf[nV] - v(i, j, nZ + rad - nH - 1, nV) / sJ);
      }
    }
  }
};

// Face outflow conditions
template <class DataViewType>
struct FaceBcWest_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcWest_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_DM_DECL(j, k), 0, nQ, nV) = v(KFVM_DM_DECL(j, k), 1, nQ, nV);
      }
    }
  }
};

template <class DataViewType>
struct FaceBcEast_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcEast_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_DM_DECL(j, k), 1, nQ, nV) = v(KFVM_DM_DECL(j, k), 0, nQ, nV);
      }
    }
  }
};

template <class DataViewType>
struct FaceBcSouth_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcSouth_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_DM_DECL(i, k), 0, nQ, nV) = v(KFVM_DM_DECL(i, k), 1, nQ, nV);
      }
    }
  }
};

template <class DataViewType>
struct FaceBcNorth_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcNorth_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuad = SPACE_DIM == 2 ? NUM_QUAD_PTS : NUM_QUAD_PTS * NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < nQuad; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_DM_DECL(i, k), 1, nQ, nV) = v(KFVM_DM_DECL(i, k), 0, nQ, nV);
      }
    }
  }
};

template <class DataViewType>
struct FaceBcBottom_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcBottom_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, 0, nQ, nV) = v(i, j, 1, nQ, nV);
      }
    }
  }
};

template <class DataViewType>
struct FaceBcTop_K<DataViewType, BCType::outflow> {
  DataViewType v;

  FaceBcTop_K(const DataViewType &v_) : v(v_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS * NUM_QUAD_PTS; nQ++) {
      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, 1, nQ, nV) = v(i, j, 0, nQ, nV);
      }
    }
  }
};

// Cell reflecting conditions
template <class DataViewType, class RefCoeff>
struct CellBcWest_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nX;

  CellBcWest_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const RefCoeff &pf_, idx_t rad_, idx_t nX_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nX(nX_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(rad + nH, j, k));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(rad + nH, j, k), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(rad - nH - 1, j, k));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(rad - nH - 1, j, k), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(KFVM_D_DECL(rad + nH, j, k), nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(rad - nH - 1, j, k), nV) = dJ * vars[nV];
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct CellBcEast_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nX;

  CellBcEast_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const RefCoeff &pf_, idx_t rad_, idx_t nX_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nX(nX_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(nX + rad - nH - 1, j, k));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(nX + rad - nH - 1, j, k), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(nX + rad + nH, j, k));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(nX + rad + nH, j, k), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(KFVM_D_DECL(nX + rad - nH - 1, j, k), nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(nX + rad + nH, j, k), nV) = dJ * vars[nV];
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct CellBcSouth_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nY;

  CellBcSouth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const RefCoeff &pf_, idx_t rad_, idx_t nY_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nY(nY_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, rad + nH, k));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(i, rad + nH, k), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, rad - nH - 1, k));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(i, rad - nH - 1, k), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(KFVM_D_DECL(i, rad + nH, k), nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, rad - nH - 1, k), nV) = dJ * vars[nV];
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct CellBcNorth_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nY;

  CellBcNorth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const RefCoeff &pf_, idx_t rad_, idx_t nY_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nY(nY_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, nY + rad - nH - 1, k));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(i, nY + rad - nH - 1, k), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, nY + rad + nH, k));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(i, nY + rad + nH, k), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(KFVM_D_DECL(i, nY + rad - nH - 1, k), nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, nY + rad + nH, k), nV) = dJ * vars[nV];
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct CellBcBottom_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nZ;

  CellBcBottom_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                 const RefCoeff &pf_, idx_t rad_, idx_t nZ_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nZ(nZ_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, j, rad + nH));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(i, j, rad + nH), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, j, rad - nH - 1));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(i, j, rad - nH - 1), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(i, j, rad + nH, nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, rad - nH - 1, nV) = dJ * vars[nV];
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct CellBcTop_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  RefCoeff pf;
  idx_t rad, nZ;

  CellBcTop_K(const DataViewType &v_, const Geometry<geomType> &geom_,
              const RefCoeff &pf_, idx_t rad_, idx_t nZ_)
      : v(v_), geom(geom_), pf(pf_), rad(rad_), nZ(nZ_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get Jacobians at source and target points
      const Real sJ = geom.jacobian(KFVM_D_DECL(i, j, nZ + rad - nH - 1));
      const Mat3 sMD =
          geom.metricDerivs(KFVM_D_DECL(i, j, nZ + rad - nH - 1), KFVM_D_DECL(0, 0, 0));
      const Real dJ = geom.jacobian(KFVM_D_DECL(i, j, nZ + rad + nH));
      const Mat3 dMD =
          geom.invMetricDerivs(KFVM_D_DECL(i, j, nZ + rad + nH), KFVM_D_DECL(0, 0, 0));

      // Extract variables from source cell, remove J factor
      Real vars[NUM_VARS];
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] = v(i, j, nZ + rad - nH - 1, nV) / sJ;
      }

      // Convert vector valued quantities to computational space
      Physics::CompSpaceVars<eqType>::convert(sMD, vars);

      // Reflect needed components
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        vars[nV] *= pf(nV);
      }

      // Convert back to physical quantities
      Physics::CompSpaceVars<eqType>::convert(dMD, vars);

      // Apply destination jacobian and store
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, nZ + rad + nH, nV) = dJ * vars[nV];
      }
    }
  }
};

// Face reflecting conditions
template <class DataViewType, class RefCoeff>
struct FaceBcWest_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;

  FaceBcWest_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const QuadRuleView &ab_, const RefCoeff &pf_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(0, j, k), KFVM_D_DECL(-0.5, ab(nQ), ab(nR)));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(0, j, k), KFVM_D_DECL(-0.5, ab(nQ), ab(nR)));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(KFVM_DM_DECL(j, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(KFVM_DM_DECL(j, k), 0, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct FaceBcEast_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;
  idx_t nX;

  FaceBcEast_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const QuadRuleView &ab_, const RefCoeff &pf_, const idx_t nX_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_), nX(nX_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(nX, j, k), KFVM_D_DECL(0.5, ab(nQ), ab(nR)));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(nX, j, k), KFVM_D_DECL(0.5, ab(nQ), ab(nR)));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(KFVM_DM_DECL(j, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(KFVM_DM_DECL(j, k), 1, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct FaceBcSouth_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;

  FaceBcSouth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const QuadRuleView &ab_, const RefCoeff &pf_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(i, 0, k), KFVM_D_DECL(ab(nQ), -0.5, ab(nR)));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(i, 0, k), KFVM_D_DECL(ab(nQ), -0.5, ab(nR)));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(KFVM_DM_DECL(i, k), 1, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(KFVM_DM_DECL(i, k), 0, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct FaceBcNorth_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;
  idx_t nY;

  FaceBcNorth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const QuadRuleView &ab_, const RefCoeff &pf_, const idx_t nY_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_), nY(nY_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
    // Loop over quadrature points
    const idx_t nQuadZ = SPACE_DIM == 2 ? 1 : NUM_QUAD_PTS;
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < nQuadZ; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(i, nY, k), KFVM_D_DECL(ab(nQ), 0.5, ab(nR)));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(i, nY, k), KFVM_D_DECL(ab(nQ), 0.5, ab(nR)));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(KFVM_DM_DECL(i, k), 0, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(KFVM_DM_DECL(i, k), 1, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct FaceBcBottom_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;

  FaceBcBottom_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                 const QuadRuleView &ab_, const RefCoeff &pf_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(i, j, 0), KFVM_D_DECL(ab(nQ), ab(nR), -0.5));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(i, j, 0), KFVM_D_DECL(ab(nQ), ab(nR), -0.5));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(i, j, 1, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, 0, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

template <class DataViewType, class RefCoeff>
struct FaceBcTop_K<DataViewType, BCType::reflecting, RefCoeff> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  RefCoeff pf;
  idx_t nZ;

  FaceBcTop_K(const DataViewType &v_, const Geometry<geomType> &geom_,
              const QuadRuleView &ab_, const RefCoeff &pf_, const idx_t nZ_)
      : v(v_), geom(geom_), ab(ab_), pf(pf_), nZ(nZ_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Get Jacobians at source and target points
        const Mat3 sMD =
            geom.metricDerivs(KFVM_D_DECL(i, j, nZ), KFVM_D_DECL(ab(nQ), ab(nR), 0.5));
        const Mat3 dMD =
            geom.invMetricDerivs(KFVM_D_DECL(i, j, nZ), KFVM_D_DECL(ab(nQ), ab(nR), 0.5));

        // Extract variables from source cell, remove J factor
        Real vars[NUM_VARS];
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] = v(i, j, 0, nR * NUM_QUAD_PTS + nQ, nV);
        }

        // Convert vector valued quantities to computational space
        Physics::CompSpaceVars<eqType>::convert(sMD, vars);

        // Reflect needed components
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          vars[nV] *= pf(nV);
        }

        // Convert back to physical quantities
        Physics::CompSpaceVars<eqType>::convert(dMD, vars);

        // Store reflected variables
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, 1, nR * NUM_QUAD_PTS + nQ, nV) = vars[nV];
        }
      }
    }
  }
};

// User conditions
template <class DataViewType>
struct CellBcWest_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nX;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcWest_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
               idx_t nX_, Real t_, EosParameters &eosParams_, UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nX(nX_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
#if (SPACE_DIM == 2)
    const idx_t k = 0;
#endif
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(-nH - 1, j - rad, k - rad));

      // query user defined BCs
      Real state[NUM_VARS];
      CellWest(v, rad, nX, j, k, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
               state);

      // Copy idx_to view
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(rad - nH - 1, j, k), nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcWest_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcWest_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const QuadRuleView &ab_, Real t_, EosParameters &eosParams_,
               UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      // Coordinate in physical space
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(0, j, 0), KFVM_D_DECL(-0.5, ab(nQ), 0.0));

      // query user defined BCs
      Real state[NUM_VARS];
      FaceWest(v, j, 0, nQ, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams, state);

      // Copy state over
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(j, 0, nQ, nV) = state[nV];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t j, const idx_t k) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(0, j, k), KFVM_D_DECL(-0.5, ab(nQ), ab(nR)));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceWest(v, j, k, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                 userParams, state);

        // Copy state over
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(j, k, 0, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

template <class DataViewType>
struct CellBcEast_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nX;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcEast_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
               idx_t nX_, Real t_, EosParameters &eosParams_, UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nX(nX_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t j, const idx_t k)) const {
#if (SPACE_DIM == 2)
    const idx_t k = 0;
#endif
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(nX + nH, j - rad, k - rad));

      // query user defined BCs
      Real state[NUM_VARS];
      CellEast(v, rad, nX, j, k, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
               state);

      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(nX + rad + nH, j, k), nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcEast_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  idx_t nX;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcEast_K(const DataViewType &v_, const Geometry<geomType> &geom_,
               const QuadRuleView &ab_, idx_t nX_, Real t_, EosParameters &eosParams_,
               UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), nX(nX_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      // Coordinate in physical space
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(nX - 1, j, 0), KFVM_D_DECL(0.5, ab(nQ), 0.0));

      // query user defined BCs
      Real state[NUM_VARS];
      FaceEast(v, j, 0, nQ, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams, state);

      // Copy state over
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(j, 1, nQ, nV) = state[nV];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t j, const idx_t k) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(nX - 1, j, k), KFVM_D_DECL(0.5, ab(nQ), ab(nR)));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceEast(v, j, k, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                 userParams, state);

        // Copy state over
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(j, k, 1, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

template <class DataViewType>
struct CellBcSouth_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nY;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcSouth_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                idx_t nY_, Real t_, EosParameters &eosParams_,
                UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nY(nY_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
#if (SPACE_DIM == 2)
    const idx_t k = 0;
#endif
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i - rad, -nH - 1, k - rad));

      // query user defined BCs
      Real state[NUM_VARS];
      CellSouth(v, rad, nY, i, k, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
                state);

      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, rad - nH - 1, k), nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcSouth_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcSouth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const QuadRuleView &ab_, Real t_, EosParameters &eosParams_,
                UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      // Coordinate in physical space
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i, 0, 0), KFVM_D_DECL(ab(nQ), -0.5, 0.0));

      // query user defined BCs
      Real state[NUM_VARS];
      FaceSouth(v, i, 0, nQ, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams, state);

      // Copy idx_to view
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, 0, nQ, nV) = state[nV];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t k) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(i, 0, k), KFVM_D_DECL(ab(nQ), -0.5, ab(nR)));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceSouth(v, i, k, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                  userParams, state);

        // Copy idx_to view
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, k, 0, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

template <class DataViewType>
struct CellBcNorth_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nY;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcNorth_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                idx_t nY_, Real t_, EosParameters &eosParams_,
                UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nY(nY_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(KFVM_DM_DECL(const idx_t i, const idx_t k)) const {
#if (SPACE_DIM == 2)
    const idx_t k = 0;
#endif
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i - rad, nY + nH, k - rad));

      // query user defined BCs
      Real state[NUM_VARS];
      CellNorth(v, rad, nY, i, k, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
                state);

      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(KFVM_D_DECL(i, nY + rad + nH, k), nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcNorth_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  idx_t nY;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcNorth_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                const QuadRuleView &ab_, idx_t nY_, Real t_, EosParameters &eosParams_,
                UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), nY(nY_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      // Coordinate in physical space
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i, nY - 1, 0), KFVM_D_DECL(ab(nQ), 0.5, 0.0));

      // query user defined BCs
      Real state[NUM_VARS];
      FaceNorth(v, i, 0, nQ, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams, state);

      // copy state over
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, 1, nQ, nV) = state[nV];
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t k) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(i, nY - 1, k), KFVM_D_DECL(ab(nQ), 0.5, ab(nR)));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceNorth(v, i, k, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                  userParams, state);

        // copy state over
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, k, 1, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

template <class DataViewType>
struct CellBcBottom_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nZ;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcBottom_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
                 idx_t nZ_, Real t_, EosParameters &eosParams_,
                 UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nZ(nZ_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i - rad, j - rad, -nH - 1));

      // query user defined BCs
      Real state[NUM_VARS];
      CellBottom(v, rad, nZ, i, j, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
                 state);

      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, rad - nH - 1, nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcBottom_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcBottom_K(const DataViewType &v_, const Geometry<geomType> &geom_,
                 const QuadRuleView &ab_, Real t_, EosParameters &eosParams_,
                 UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(i, j, 0), KFVM_D_DECL(ab(nQ), ab(nR), -0.5));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceBottom(v, i, j, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                   userParams, state);

        // copy state over
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, 0, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

template <class DataViewType>
struct CellBcTop_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  idx_t rad, nZ;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  CellBcTop_K(const DataViewType &v_, const Geometry<geomType> &geom_, idx_t rad_,
              idx_t nZ_, Real t_, EosParameters &eosParams_, UserParameters &userParams_)
      : v(v_), geom(geom_), rad(rad_), nZ(nZ_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over halo
    for (idx_t nH = 0; nH < rad; nH++) {
      // Get cell center in halo
      Vec3 xyz = geom.physCoord(KFVM_D_DECL(i - rad, j - rad, nZ + nH));

      // query user defined BCs
      Real state[NUM_VARS];
      CellTop(v, rad, nZ, i, j, nH, xyz.v1, xyz.v2, xyz.v3, t, eosParams, userParams,
              state);

      // Loop over SimVars
      for (idx_t nV = 0; nV < NUM_VARS; nV++) {
        v(i, j, nZ + rad + nH, nV) = state[nV];
      }
    }
  }
};

template <class DataViewType>
struct FaceBcTop_K<DataViewType, BCType::user> {
  DataViewType v;
  Geometry<geomType> geom;
  QuadRuleView ab;
  idx_t nZ;
  Real t;
  EosParameters eosParams;
  UserParameters userParams;

  FaceBcTop_K(const DataViewType &v_, const Geometry<geomType> &geom_,
              const QuadRuleView &ab_, idx_t nZ_, Real t_, EosParameters &eosParams_,
              UserParameters &userParams_)
      : v(v_), geom(geom_), ab(ab_), nZ(nZ_), t(t_), eosParams(eosParams_),
        userParams(userParams_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const idx_t i, const idx_t j) const {
    // Loop over quadrature points
    for (idx_t nQ = 0; nQ < NUM_QUAD_PTS; nQ++) {
      for (idx_t nR = 0; nR < NUM_QUAD_PTS; nR++) {
        // Coordinate in physical space
        Vec3 xyz =
            geom.physCoord(KFVM_D_DECL(i, j, nZ - 1), KFVM_D_DECL(ab(nQ), ab(nR), 0.5));

        // query user defined BCs
        Real state[NUM_VARS];
        FaceTop(v, i, j, nQ * NUM_QUAD_PTS + nR, xyz.v1, xyz.v2, xyz.v3, t, eosParams,
                userParams, state);

        // copy state over
        for (idx_t nV = 0; nV < NUM_VARS; nV++) {
          v(i, j, 1, nQ * NUM_QUAD_PTS + nR, nV) = state[nV];
        }
      }
    }
  }
};

} // end namespace BoundaryConditions

} // end namespace KFVM

#endif
