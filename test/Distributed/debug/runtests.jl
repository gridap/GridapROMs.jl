using Test

@testset "distributed poisson (debug)" begin include("PoissonDistributed.jl") end
@testset "distributed stokes (debug)" begin include("StokesDistributed.jl") end
