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

template <GeometryType gt>
struct Geometry {
};

// Specialization for cartesian geometry
// Simplest case, here computational and physical space are the same
template <>
struct Geometry<GeometryType::Cartesian> {
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

  // Convenience functions to calculate coordinates at cell centers
  KOKKOS_INLINE_FUNCTION
  Real xiCoord(const idx_t n) const { return dXi * (Real(n) + Real(0.5)) + xiLo; }

  KOKKOS_INLINE_FUNCTION
  Real etaCoord(const idx_t n) const { return dEta * (Real(n) + Real(0.5)) + etaLo; }

  KOKKOS_INLINE_FUNCTION
  Real zetaCoord(const idx_t n) const { return dZeta * (Real(n) + Real(0.5)) + zetaLo; }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(const Real xi, const Real eta, const Real zeta) const { return 1.0; }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    return 1.0;
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const Real xi, const Real eta, const Real zeta) const {
    return {xi, eta, zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j) const {
    const Real xi = xiCoord(i), eta = etaCoord(j);
    return {xi, eta, zetaLo};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const idx_t k) const {
    const Real xi = xiCoord(i), eta = etaCoord(j);
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k);
    return {xi, eta, zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const Real abXi, const Real abEta) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {xi, eta, zetaLo};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const idx_t k, const Real abXi,
                 const Real abEta, const Real abZeta) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k) + dZeta * abZeta;
    return {xi, eta, zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 physCoordJac(const idx_t i, const idx_t j, const idx_t k, const Real abXi,
                    const Real abEta, const Real abZeta) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k) + dZeta * abZeta;
    const Real J = jacobian(xi, eta, zeta);
    return {xi, eta, zeta, J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 xiMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                     KFVM_D_DECL(const Real abXi, const Real abEta,
                                 const Real abZeta)) const {
    return {1.0, 0.0, 0.0, 1.0};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 etaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                      KFVM_D_DECL(const Real abXi, const Real abEta,
                                  const Real abZeta)) const {
    return {0.0, 1.0, 0.0, 1.0};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 zetaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                       KFVM_D_DECL(const Real abXi, const Real abEta,
                                   const Real abZeta)) const {
    return {0.0, 0.0, 1.0, 1.0};
  }

  Geometry() = default;
};

// Specialization for cartesian geometry
// Simplest case, here computational and physical space are the same
template <>
struct Geometry<GeometryType::Polar> {
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

  // Convenience functions to calculate coordinates at cell centers
  KOKKOS_INLINE_FUNCTION
  Real xiCoord(const idx_t n) const { return dXi * (Real(n) + Real(0.5)) + xiLo; }

  KOKKOS_INLINE_FUNCTION
  Real etaCoord(const idx_t n) const { return dEta * (Real(n) + Real(0.5)) + etaLo; }

  KOKKOS_INLINE_FUNCTION
  Real zetaCoord(const idx_t n) const { return dZeta * (Real(n) + Real(0.5)) + zetaLo; }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(const Real xi, const Real eta, const Real zeta) const { return xi; }

  KOKKOS_INLINE_FUNCTION
  Real jacobian(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k)) const {
    return xiCoord(i);
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const Real xi, const Real eta, const Real zeta) const {
    using Kokkos::cos;
    using Kokkos::sin;
    return {xi * cos(eta), xi * sin(eta), zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i), eta = etaCoord(j);
    return {xi * cos(eta), xi * sin(eta), zetaLo};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const idx_t k) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i), eta = etaCoord(j);
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k);
    return {xi * cos(eta), xi * sin(eta), zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const Real abXi, const Real abEta) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {xi * cos(eta), xi * sin(eta), zetaLo};
  }

  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const idx_t k, const Real abXi,
                 const Real abEta, const Real abZeta) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k) + dZeta * abZeta;
    return {xi * cos(eta), xi * sin(eta), zeta};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 physCoordJac(const idx_t i, const idx_t j, const Real abXi,
                    const Real abEta) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = zetaLo;
    const Real J = jacobian(xi, eta, zeta);
    return {xi * cos(eta), xi * sin(eta), zeta, J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 physCoordJac(const idx_t i, const idx_t j, const idx_t k, const Real abXi,
                    const Real abEta, const Real abZeta) const {
    using Kokkos::cos;
    using Kokkos::sin;
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = SPACE_DIM == 2 ? zetaLo : zetaCoord(k) + dZeta * abZeta;
    const Real J = jacobian(xi, eta, zeta);
    return {xi * cos(eta), xi * sin(eta), zeta, J};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 xiMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                     KFVM_D_DECL(const Real abXi, const Real abEta,
                                 const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {cos(eta), sin(eta), 0.0, xi};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 etaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                      KFVM_D_DECL(const Real abXi, const Real abEta,
                                  const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {-sin(eta) / xi, cos(eta) / xi, 0.0, xi};
  }

  KOKKOS_INLINE_FUNCTION
  Vec4 zetaMetricDeriv(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                       KFVM_D_DECL(const Real abXi, const Real abEta,
                                   const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi;
    return {0.0, 0.0, 1.0, xi};
  }

  KOKKOS_INLINE_FUNCTION
  Mat3 metricDerivs(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                    KFVM_D_DECL(const Real abXi, const Real abEta,
                                const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {cos(eta), sin(eta), 0.0, -sin(eta) / xi, cos(eta) / xi, 0.0, 0.0, 0.0, 1.0};
  }

  KOKKOS_INLINE_FUNCTION
  Mat3 invMetricDerivs(KFVM_D_DECL(const idx_t i, const idx_t j, const idx_t k),
                       KFVM_D_DECL(const Real abXi, const Real abEta,
                                   const Real abZeta)) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    return {cos(eta), -xi * sin(eta), 0.0, sin(eta), xi * cos(eta), 0.0, 0.0, 0.0, 1.0};
  }

  Geometry() = default;
};

} // namespace KFVM

#endif
