module HeatEqDistributedMPI

using PartitionedArrays
include("../HeatEqDistributed.jl")

with_mpi() do distribute
  HeatEqDistributed.main(distribute,(2,2))
end

end