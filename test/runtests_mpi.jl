using Test

# Runs the MPI-backed distributed tests. Unlike the rest of the test suite
# (`runtests.jl`), this file must be launched through `mpiexecjl`, e.g.:
#
#   mpiexecjl -n 4 julia --project=test test/runtests_mpi.jl
#
# `mpiexecjl` is installed by `MPI.install_mpiexecjl()`; see the MPI.jl docs.
# Each file under `Distributed/mpi/` is itself re-launched as a subprocess
# with `MPI.mpiexec()`, so this driver only needs to be started once.

include("Distributed/mpi/runtests.jl")
