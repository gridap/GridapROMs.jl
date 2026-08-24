module PoissonDistributedDebug

using PartitionedArrays
include("../PoissonDistributed.jl")

with_debug() do distribute
  PoissonDistributed.main(distribute,(2,2))
end

end # module
