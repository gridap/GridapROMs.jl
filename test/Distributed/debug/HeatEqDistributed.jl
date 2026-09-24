module HeatEqDistributedDebug

using PartitionedArrays
include("../HeatEqDistributed.jl")

with_debug() do distribute
  HeatEqDistributed.main(distribute,(2,2))
end

end