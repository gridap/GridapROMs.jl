module PoissonDistributedMPI

using PartitionedArrays
include("../PoissonDistributed.jl")

# with_mpi() do distribute
#   for compression in (:local,:global), hypred_strategy in (:deim,:sopt,)
#     PoissonDistributed.main(distribute,(2,2),compression,hypred_strategy)
#   end
# end

end
