module StokesDistributedMPI

using PartitionedArrays
include("../StokesDistributed.jl")

with_mpi() do distribute
  for compression in (:local,:global), hypred_strategy in (:deim,)
    StokesDistributed.main(distribute,(2,2),compression,hypred_strategy)
  end
end

end
