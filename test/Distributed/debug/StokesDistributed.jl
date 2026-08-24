module StokesDistributedDebug

using PartitionedArrays
include("../StokesDistributed.jl")

with_debug() do distribute
  StokesDistributed.main(distribute,(2,2))
end

end # module
