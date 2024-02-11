# KFVM-Kokkos

KFVM-Kokkos is a finite volume code utilizing kernel-based reconstruction, and is parallelized via Kokkos.

# Dependencies

## Required

* CMake 3.16+
* C++17 compiler
* MPI
* HDF5
* Kokkos (included as submodule)
* Kokkos-Kernels (included as submodule)
* PDI (included as submodule)
* fmt (included as submodule)

## Optional (Python + modules)

* python3
* questionary (for problem generator script)
* numpy (for plotting and error calculators)
* h5py (for plotting and error calculators)
* matplotlib (for plotting)

# Getting started

## Initializing submodules and building dependencies

The submodules can be cloned by running

	git submodule init
	git submodule update
	
and all appear inside the ``external`` subdirectory
	
### Building Kokkos and Kokkos-Kernels

Refer to the [Kokkos wiki](https://kokkos.github.io/kokkos-core-wiki/building.html#configuring-cmake) for directions on building Kokkos using CMake.

Refer to the [Kokkos-Kernels repository](https://github.com/kokkos/kokkos-kernels#cmake) for directions on building Kokkos-Kernels.

### Building PDI

The default configuration choices for PDI are satisfactory. One can simply execute:

	cd external/pdi
	mkdir build && cd build
	cmake ..
	make -j
	sudo make install
	
Refer to the [PDI documentation](https://pdi.dev/master/index.html) for more information if interested.

### Building fmt

The default configuration choices for fmt are satisfactory. One can simply execute:

	cd external/fmt
	mkdir build && cd build
	cmake ..
	make -j
	sudo make install
	
Refer to the [fmt documentation](https://fmt.dev/latest/index.html) for more information if interested.

## Running included sample problems

Sample problems for each supported set of equations are available in the ``exec`` directory. You will need to call ``cmake`` inside each one to build an executable for that problem. To simplify this process it is useful to export both the root directory of KFVM-Kokkos as a persistant shell variable, and create an alias for the specific cmake command needed for your system. For example, you could have

	export KFVM_KOKKOS_ROOT=$HOME/Code/kfvm-kokkos
	alias kfvm-cmake='cmake $KFVM_KOKKOS_ROOT -DCMAKE_CXX_COMPILER=nvcc_wrapper'
	
With this you would cd into one of the sample problem directories and compile there like so:

	cd /exec/Hydro/Sod # or whichever problem directory
	kfvm-cmake
	make -j
	
There should now be an executable called ``kfvm.ex`` in the ``bin`` directory. Running the code is accomplished via

	bin/kfvm.ex sod.init
	
where the initialization file holds all run time configurations.

## Adding your own problems

New problems should be added somewhere outside of the KFVM-Kokkos directory. If you have python3 and questionary available on your system, the script ``python/problemGenerator.py`` can be used to interactively create all needed files for a new problem. Otherwise, one can just mimic the layout of any of the included problems.

# Forthcoming documentation

This code exists to support the owner's PhD work. More detailed documentation will hopefully be produced later on, but is not presently a priority.

# Caveats

Not everything in this code is fully tested, or necessarily in its final state. In particular:

* Special relativistic hydrodynamics support is present in the code, but do not yet yield useful results
* Eight-wave MHD is mostly an afterthought from the GLM implementation, and there is an unfortunate amount of code duplication there

On the other hand, the Euler and Navier-Stokes equations are well supported and mostly stable. GLM based MHD is also nearly stable.
