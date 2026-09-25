using Test

@testset "distributed poisson (debug)" begin include("PoissonDistributed.jl") end
@testset "distributed heat equation (debug)" begin include("HeatEqDistributed.jl") end
@testset "distributed stokes (debug)" begin include("StokesDistributed.jl") end
@testset "distributed transient stokes (debug)" begin include("TransientStokesDistributed.jl") end
