#ifndef PRINTERMPI_H_
#define PRINTERMPI_H_

#include <cstdarg>

#include <fmt/color.h>
#include <fmt/core.h>

#include "ProblemSetup.H"

namespace KFVM {

namespace Print {

template <typename... T>
void Single(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void SingleV1(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0 && ps.verbose >= 1) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void SingleV2(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0 && ps.verbose >= 2) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void SingleV3(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0 && ps.verbose >= 3) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void Any(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  fmt::print(fstr, args...);
}

template <typename... T>
void AnyV1(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.verbose >= 1) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void AnyV2(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.verbose >= 2) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void AnyV3(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.verbose >= 3) {
    fmt::print(fstr, args...);
  }
}

template <typename... T>
void AlertSingle(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0) {
    std::string str = fmt::format(fstr, args...);
    fmt::print(fmt::fg(fmt::color::golden_rod), str);
  }
}

template <typename... T>
void AlertAny(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  std::string str = fmt::format(fstr, args...);
  fmt::print(fmt::fg(fmt::color::golden_rod), str);
}

template <typename... T>
void WarnSingle(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  if (ps.layoutMPI.rank == 0) {
    std::string str = fmt::format(fstr, args...);
    fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), str);
  }
}

template <typename... T>
void WarnAny(const ProblemSetup &ps, fmt::format_string<T...> fstr, T &&...args) {
  std::string str = fmt::format(fstr, args...);
  fmt::print(fmt::emphasis::bold | fmt::fg(fmt::color::red), str);
}

} // end namespace Print
} // end namespace KFVM

#endif
