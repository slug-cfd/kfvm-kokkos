#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

struct hello_world {
  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // FIXME_SYCL needs workaround for printf
#ifndef __SYCL_DEVICE_ONLY__
    printf("Hello from i = %i\n", i);
#else
    (void)i;
#endif
  }
};

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  
  printf("Hello World on Kokkos execution space %s\n",typeid(Kokkos::DefaultExecutionSpace).name());
  Kokkos::parallel_for("HelloWorld", 15, hello_world());
  
  Kokkos::finalize();
}
