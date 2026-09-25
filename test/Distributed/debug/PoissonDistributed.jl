module PoissonDistributedDebug

using PartitionedArrays
include("../PoissonDistributed.jl")

with_debug() do distribute
  for compression in (:local,:global), hypred_strategy in (:deim,:sopt,:rbf,:none,:affine)
    PoissonDistributed.main(distribute,(2,2),compression,hypred_strategy)
  end
end

end
