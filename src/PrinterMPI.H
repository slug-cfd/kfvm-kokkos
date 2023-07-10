#ifndef PRINTERMPI_H_
#define PRINTERMPI_H_

#include <cstdarg>
#include <cstdio>

#include "ProblemSetup.H"

namespace KFVM {

void PrintSingle(const ProblemSetup &ps, const char *fmt, ...) {
  if (ps.layoutMPI.rank == 0) {
    std::va_list args;
    va_start(args, fmt);
    std::vprintf(fmt, args);
    va_end(args);
  }
}

void PrintAll(const ProblemSetup &ps, const char *fmt, ...) {
  std::va_list args;
  va_start(args, fmt);
  std::vprintf(fmt, args);
  va_end(args);
}

} // end namespace KFVM

#endif
