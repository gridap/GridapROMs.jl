module MulMPI

using MPI, PartitionedArrays
include("../mul.jl")

with_mpi() do distribute
  Mul.main(distribute,(4,))
end

end
