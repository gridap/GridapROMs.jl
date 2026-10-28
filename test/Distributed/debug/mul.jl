module MulDebug

using PartitionedArrays
include("../mul.jl")

# NOTE: smoke test only. `Mul.main`'s test6/test9 (see ../mul.jl) are
# documented to give a false pass under DebugArray, since every "rank" there
# shares the same underlying memory, so there is no real cross-rank aliasing
# for them to catch. The correctness check for those paths only exists under
# real MPI: see ../mpi/mul.jl.
with_debug() do distribute
  Mul.main(distribute,(4,))
end

end
