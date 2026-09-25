module HeatEqDistributedMPI

using PartitionedArrays
include("../HeatEqDistributed.jl")

with_mpi() do distribute
  for compression in (:local,:global), hypred_strategy in (:deim,:sopt,:rbf,:none,:affine)
    HeatEqDistributed.main(distribute,(2,2),compression,hypred_strategy)
  end
end

end
