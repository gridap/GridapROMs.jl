module PoissonDistributedMPI

using MPI, PartitionedArrays
include("../PoissonDistributed.jl")

with_mpi() do distribute
  PoissonDistributed.main(distribute,(2,2))
end

end
