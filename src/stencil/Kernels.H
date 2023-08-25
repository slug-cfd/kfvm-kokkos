#ifndef KERNELS_H_
#define KERNELS_H_

#include "../Types.H"
#include "KernelTypes.H"

namespace KFVM {

namespace Stencil {

template <KernelType kt>
struct Kernel {};

} // namespace Stencil

} // namespace KFVM

#include "SEKernel.H"

#include "PHSKernel.H"

#endif
