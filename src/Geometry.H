#ifndef GEOMETRY_H_
#define GEOMETRY_H_

// File: Geometry.H
// Purpose: Hold grid geometry and convenience functions

#include <cstdio>

#include <Kokkos_Core.hpp>

#include <Definitions.H>

#include "Dimension.H"
#include "GeometryTypes.H"
#include "ProblemSetup.H"
#include "Types.H"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace KFVM {

struct Mesh {
  MeshDataView x, y, z;
  Mesh(const idx_t nX, const idx_t nY, const idx_t nZ)
      : x("meshx", nX, nY, nZ), y("meshy", nX, nY, nZ), z("meshz", nX, nY, nZ) {}
};

// Base geometry struct with all common routines
// The first few stubbed out functions must be specialized for each geometry type
template <GeometryType gt>
struct Geometry {
  // Grid spacing (computational domain)
  const Real dXi, dEta, dZeta, dmin;

  // Bounding box (computational domain)
  const Real xiLo, xiUp, etaLo, etaUp, zetaLo, zetaUp;

  // Construct from Problemsetup object
  Geometry(const ProblemSetup &ps)
      : dXi((ps.xUp_g - ps.xLo_g) / (ps.nX * ps.nbX)),
        dEta((ps.yUp_g - ps.yLo_g) / (ps.nY * ps.nbY)),
        dZeta((ps.zUp_g - ps.zLo_g) / (ps.nZ * ps.nbZ)),
        xiLo(ps.xLo_g + dXi * ps.layoutMPI.bxLo),
        xiUp(ps.xLo_g + dXi * ps.layoutMPI.bxUp),
        etaLo(ps.yLo_g + dEta * ps.layoutMPI.byLo),
        etaUp(ps.yLo_g + dEta * ps.layoutMPI.byUp),
        zetaLo(ps.zLo_g + dZeta * ps.layoutMPI.bzLo),
        zetaUp(ps.zLo_g + dZeta * ps.layoutMPI.bzUp),
#if (SPACE_DIM == 2)
        dmin(std::fmin(dXi, dEta))
#else
        dmin(std::fmin(dXi, std::fmin(dEta, dZeta)))
#endif
  {
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Real jacobian(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Vec3 xMetricDeriv(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Vec3 yMetricDeriv(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Vec3 zMetricDeriv(const Real xi, const Real eta, const Real zeta) const;

  // Convenience functions to calculate coordinates at cell centers
  KOKKOS_INLINE_FUNCTION
  Real xiCoord(const idx_t n) const { return dXi * (Real(n) + Real(0.5)) + xiLo; }

  KOKKOS_INLINE_FUNCTION
  Real etaCoord(const idx_t n) const { return dEta * (Real(n) + Real(0.5)) + etaLo; }

  KOKKOS_INLINE_FUNCTION
  Real zetaCoord(const idx_t n) const { return dZeta * (Real(n) + Real(0.5)) + zetaLo; }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    const Real xi = xiCoord(i), eta = etaCoord(j);
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k);
#endif
    return jacobian(xi, eta, zeta);
  }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                KFVM_D_DECL(const Real abXi, const Real abEta, const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    return jacobian(xi, eta, zeta);
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    const Real xi = xiCoord(i), eta = etaCoord(j);
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k);
#endif
    return physCoord(xi, eta, zeta);
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                 KFVM_D_DECL(const Real abXi, const Real abEta,
                             const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    return physCoord(xi, eta, zeta);
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 physCoordJac(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                    KFVM_D_DECL(const Real abXi, const Real abEta,
                                const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 pc = physCoord(xi, eta, zeta);
    const Real J = jacobian(xi, eta, zeta);
    return {pc.v1, pc.v2, pc.v3, J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 xiMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                     KFVM_D_DECL(const Real abXi, const Real abEta,
                                 const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 xR = xMetricDeriv(xi, eta, zeta);
    const Vec3 yR = yMetricDeriv(xi, eta, zeta);
    const Vec3 zR = zMetricDeriv(xi, eta, zeta);
    const Real J = jacobian(xi, eta, zeta);
    return {(yR.v2 * zR.v3 - yR.v3 * zR.v2) / J, (xR.v3 * zR.v2 - xR.v2 * zR.v3) / J,
            (xR.v2 * yR.v3 - xR.v3 * yR.v2) / J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 etaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                      KFVM_D_DECL(const Real abXi, const Real abEta,
                                  const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 xR = xMetricDeriv(xi, eta, zeta);
    const Vec3 yR = yMetricDeriv(xi, eta, zeta);
    const Vec3 zR = zMetricDeriv(xi, eta, zeta);
    const Real J = jacobian(xi, eta, zeta);
    return {(yR.v3 * zR.v1 - yR.v1 * zR.v3) / J, (xR.v1 * zR.v3 - xR.v3 * zR.v1) / J,
            (xR.v3 * yR.v1 - xR.v1 * yR.v3) / J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 zetaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                       KFVM_D_DECL(const Real abXi, const Real abEta,
                                   const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 xR = xMetricDeriv(xi, eta, zeta);
    const Vec3 yR = yMetricDeriv(xi, eta, zeta);
    const Vec3 zR = zMetricDeriv(xi, eta, zeta);
    const Real J = jacobian(xi, eta, zeta);
    return {(yR.v1 * zR.v2 - yR.v2 * zR.v1) / J, (xR.v2 * zR.v1 - xR.v1 * zR.v2) / J,
            (xR.v1 * yR.v2 - xR.v2 * yR.v1) / J};
  }

  KOKKOS_INLINE_FUNCTION
  Mat3 metricDerivs(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                    KFVM_D_DECL(const Real abXi, const Real abEta,
                                const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 xR = xMetricDeriv(xi, eta, zeta);
    const Vec3 yR = yMetricDeriv(xi, eta, zeta);
    const Vec3 zR = zMetricDeriv(xi, eta, zeta);
    const Real J = jacobian(xi, eta, zeta);
    return {(yR.v2 * zR.v3 - yR.v3 * zR.v2) / J, (xR.v3 * zR.v2 - xR.v2 * zR.v3) / J,
            (xR.v2 * yR.v3 - xR.v3 * yR.v2) / J, (yR.v3 * zR.v1 - yR.v1 * zR.v3) / J,
            (xR.v1 * zR.v3 - xR.v3 * zR.v1) / J, (xR.v3 * yR.v1 - xR.v1 * yR.v3) / J,
            (yR.v1 * zR.v2 - yR.v2 * zR.v1) / J, (xR.v2 * zR.v1 - xR.v1 * zR.v2) / J,
            (xR.v1 * yR.v2 - xR.v2 * yR.v1) / J};
  }

  KOKKOS_INLINE_FUNCTION Mat3 invMetricDerivs(
      KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
      KFVM_D_DECL(const Real abXi, const Real abEta, const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
#if (SPACE_DIM == 2)
    const Real zeta = zetaLo;
#else
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
#endif
    const Vec3 xR = xMetricDeriv(xi, eta, zeta);
    const Vec3 yR = yMetricDeriv(xi, eta, zeta);
    const Vec3 zR = zMetricDeriv(xi, eta, zeta);
    return {xR.v1, xR.v2, xR.v3, yR.v1, yR.v2, yR.v3, zR.v1, zR.v2, zR.v3};
  }

  Geometry() = default;
};

// Cartesian specializations
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  return {xi, eta, zeta};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Cartesian>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  return 1.0;
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {1.0, 0.0, 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 1.0, 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

// Polar specializations
// Coordinates ordered as (r,phi,z)
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;
  return {xi * cos(eta), xi * sin(eta), zeta};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Polar>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  return xi;
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {cos(eta), -xi * sin(eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {sin(eta), xi * cos(eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

// Spherical specializations
// Coordinates ordered as (r,theta,phi)
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;
  return {xi * sin(eta) * cos(zeta), xi * sin(eta) * sin(zeta), xi * cos(eta)};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Spherical>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::sin;
  return xi * xi * sin(eta);
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {sin(eta) * cos(zeta), xi * cos(eta) * cos(zeta), -xi * sin(eta) * sin(zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {sin(eta) * sin(zeta), xi * cos(eta) * sin(zeta), xi * sin(eta) * cos(zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {cos(eta), -xi * sin(eta), 0.0};
}

} // namespace KFVM

#endif
