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
  const Real xiExt, etaExt, zetaExt;

  // Construct from Problemsetup object
  Geometry(const ProblemSetup &ps)
      : dXi((ps.xUp_g - ps.xLo_g) / (ps.nX * ps.nbX)),
        dEta((ps.yUp_g - ps.yLo_g) / (ps.nY * ps.nbY)),
#if (SPACE_DIM == 2)
        dZeta(std::fmin(dXi, dEta)),
#else
        dZeta((ps.zUp_g - ps.zLo_g) / (ps.nZ * ps.nbZ)),
#endif
        xiLo(ps.xLo_g + dXi * ps.layoutMPI.bxLo),
        xiUp(ps.xLo_g + dXi * ps.layoutMPI.bxUp),
        etaLo(ps.yLo_g + dEta * ps.layoutMPI.byLo),
        etaUp(ps.yLo_g + dEta * ps.layoutMPI.byUp),
        zetaLo(ps.zLo_g + dZeta * ps.layoutMPI.bzLo),
        zetaUp(ps.zLo_g + dZeta * ps.layoutMPI.bzUp),
        dmin(std::fmin(dXi, std::fmin(dEta, dZeta))), xiExt(xiUp - xiLo),
        etaExt(etaUp - etaLo), zetaExt(zetaUp - zetaLo) {
  }

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

  // Have special case in 2d where 3d points are actually needed
  // This is only for plotting
#if (SPACE_DIM == 2)
  KOKKOS_INLINE_FUNCTION
  Vec3 physCoord(const idx_t i, const idx_t j, const idx_t k, const Real abXi,
                 const Real abEta, const Real abZeta) const {
    const Real xi = xiCoord(i) + dXi * abXi, eta = etaCoord(j) + dEta * abEta;
    const Real zeta = zetaCoord(k) + dZeta * abZeta;
    return physCoord(xi, eta, zeta);
  }
#endif

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
    return xiMetricDeriv(xi, eta, zeta);
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
    return etaMetricDeriv(xi, eta, zeta);
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
    return zetaMetricDeriv(xi, eta, zeta);
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
    const Vec3 xiR = xiMetricDeriv(xi, eta, zeta);
    const Vec3 etaR = etaMetricDeriv(xi, eta, zeta);
    const Vec3 zetaR = zetaMetricDeriv(xi, eta, zeta);
    return {xiR.v1,  xiR.v2,   xiR.v3,   etaR.v1, etaR.v2,
            etaR.v3, zetaR.v1, zetaR.v2, zetaR.v3};
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

 private:
  // These functions need to be specialized for each geometry
  // see below.
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

  KOKKOS_INLINE_FUNCTION
  Vec3 xiMetricDeriv(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Vec3 etaMetricDeriv(const Real xi, const Real eta, const Real zeta) const;

  KOKKOS_INLINE_FUNCTION
  Vec3 zetaMetricDeriv(const Real xi, const Real eta, const Real zeta) const;
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

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::xiMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {1.0, 0.0, 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::etaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 1.0, 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cartesian>::zetaMetricDeriv(
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

  return {xi * cos(M_PI * eta), xi * sin(M_PI * eta), zeta};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Polar>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  return M_PI * xi;
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {cos(M_PI * eta), -M_PI * xi * sin(M_PI * eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {sin(M_PI * eta), M_PI * xi * cos(M_PI * eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::xiMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {cos(M_PI * eta), sin(M_PI * eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::etaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {-sin(M_PI * eta) / (M_PI * xi), cos(M_PI * eta) / (M_PI * xi), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Polar>::zetaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

// Cylindrical specializations
// Coordinates ordered as (r,z,phi), same as polar with different ordering
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {xi * cos(M_PI * zeta), xi * sin(M_PI * zeta), eta};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Cylindrical>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  return M_PI * xi;
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {cos(M_PI * zeta), 0.0, -M_PI * xi * sin(M_PI * zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {sin(M_PI * zeta), 0.0, M_PI * xi * cos(M_PI * zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 1.0, 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::xiMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {cos(M_PI * zeta), sin(M_PI * zeta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::etaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Cylindrical>::zetaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {-sin(M_PI * zeta) / (M_PI * xi), cos(M_PI * zeta) / (M_PI * xi), 0.0};
}

// Spherical specializations
// Coordinates ordered as (r,theta,phi)
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {xi * sin(M_PI * eta) * cos(M_PI * zeta),
          xi * sin(M_PI * eta) * sin(M_PI * zeta), xi * cos(M_PI * eta)};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Spherical>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::sin;

  return M_PI * xi * sin(M_PI * eta);
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {sin(M_PI * eta) * cos(M_PI * zeta),
          M_PI * xi * cos(M_PI * eta) * cos(M_PI * zeta),
          -M_PI * xi * sin(M_PI * eta) * sin(M_PI * zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {sin(M_PI * eta) * sin(M_PI * zeta),
          M_PI * xi * cos(M_PI * eta) * sin(M_PI * zeta),
          M_PI * xi * sin(M_PI * eta) * cos(M_PI * zeta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {cos(M_PI * eta), -M_PI * xi * sin(M_PI * eta), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::xiMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  return {sin(M_PI * eta) * cos(M_PI * zeta), sin(M_PI * eta) * sin(M_PI * zeta),
          cos(M_PI * eta)};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::etaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;
  using Kokkos::sqrt;

  const Real x = xi * sin(M_PI * eta) * cos(M_PI * zeta);
  const Real y = xi * sin(M_PI * eta) * sin(M_PI * zeta);
  const Real z = xi * cos(M_PI * eta);
  const Real rxy = sqrt(x * x + y * y), prsq = M_PI * (x * x + y * y + z * z);

  return {x * z / (rxy * prsq), y * z / (rxy * prsq), -rxy / prsq};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Spherical>::zetaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::sin;

  const Real x = xi * sin(M_PI * eta) * cos(M_PI * zeta);
  const Real y = xi * sin(M_PI * eta) * sin(M_PI * zeta);
  const Real prsq = M_PI * (x * x + y * y); // Ignoring z coord

  return {-y / prsq, x / prsq, 0.0};
}

// Wedge specialization
#define KFVM_GEOMETRY_WEDGE_FACTOR 6.0 // pi/6 angle for wedge
template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::physCoord(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;

  const Real r = pow(xi * xi + eta * eta, a / (2.0 * b));
  const Real th = (a * atan2(eta, xi) + M_PI) / b;

  return {r * cos(th), r * sin(th), zeta};
}

template <>
KOKKOS_INLINE_FUNCTION Real Geometry<GeometryType::Wedge>::jacobian(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;

  const Real rsq = xi * xi + eta * eta;
  const Real r = pow(rsq, a / b);

  return a * a * r / (b * b * rsq);
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::xMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;
  const Real c = KFVM_GEOMETRY_WEDGE_FACTOR + 1.0;

  const Real rsq = xi * xi + eta * eta;
  const Real r = pow(rsq, c / (2.0 * b));
  const Real th = (a * atan2(eta, xi) + M_PI) / b;

  return {a * (xi * cos(th) + eta * sin(th)) / (b * r),
          -a * (xi * sin(th) - eta * cos(th)) / (b * r), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::yMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;
  const Real c = KFVM_GEOMETRY_WEDGE_FACTOR + 1.0;

  const Real rsq = xi * xi + eta * eta;
  const Real r = pow(rsq, c / (2.0 * b));
  const Real th = (a * atan2(eta, xi) + M_PI) / b;

  return {a * (xi * sin(th) - eta * cos(th)) / (b * r),
          a * (xi * cos(th) + eta * sin(th)) / (b * r), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::zMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::xiMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;

  const Real rsq = xi * xi + eta * eta;
  const Real r = pow(rsq, a / (2.0 * b));
  const Real th = (a * atan2(eta, xi) + M_PI) / b;

  return {b * (xi * cos(th) + eta * sin(th)) / (a * r),
          b * (xi * sin(th) - eta * cos(th)) / (a * r), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::etaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  using Kokkos::cos;
  using Kokkos::pow;
  using Kokkos::sin;

  const Real a = KFVM_GEOMETRY_WEDGE_FACTOR - 1.0;
  const Real b = KFVM_GEOMETRY_WEDGE_FACTOR;

  const Real rsq = xi * xi + eta * eta;
  const Real r = pow(rsq, a / (2.0 * b));
  const Real th = (a * atan2(eta, xi) + M_PI) / b;

  return {-b * (xi * sin(th) - eta * cos(th)) / (a * r),
          b * (xi * cos(th) + eta * sin(th)) / (a * r), 0.0};
}

template <>
KOKKOS_INLINE_FUNCTION Vec3 Geometry<GeometryType::Wedge>::zetaMetricDeriv(
    const Real xi, const Real eta, const Real zeta) const {
  return {0.0, 0.0, 1.0};
}

} // namespace KFVM

#endif
