module DistributedDebugTests

using Test

@testset "mul!" begin include("mul.jl") end
@testset "poisson" begin include("PoissonDistributed.jl") end
@testset "heat equation" begin include("HeatEqDistributed.jl") end
@testset "stokes" begin include("StokesDistributed.jl") end
@testset "transient stokes" begin include("TransientStokesDistributed.jl") end

end
