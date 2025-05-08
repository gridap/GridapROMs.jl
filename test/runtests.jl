module GridapROMsTests

using Test

@testset "poisson" begin include("RBSteady/poisson.jl") end
@testset "steady stokes" begin include("RBSteady/stokes.jl") end
@testset "steady navier-stokes" begin include("RBSteady/navier_stokes.jl") end

@testset "heat equation" begin include("RBTransient/heat_equation.jl") end
@testset "unsteady elasticity" begin include("RBTransient/elasticity.jl") end
@testset "unsteady stokes" begin include("RBTransient/stokes.jl") end
@testset "unsteady navier-stokes" begin include("RBTransient/navier_stokes.jl") end

end # module
