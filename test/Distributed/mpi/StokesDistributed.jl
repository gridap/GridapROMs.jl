module StokesDistributedMPI

using MPI, PartitionedArrays
include("../StokesDistributed.jl")

with_mpi() do distribute
  StokesDistributed.main(distribute,(2,2))
end

end
