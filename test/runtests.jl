module GridapROMsTests

using Test

@testset "utils" begin include("FEM/utils.jl") end
@testset "param data structures" begin include("FEM/param_data_structures.jl") end
@testset "dof maps" begin include("FEM/dof_maps.jl") end
@testset "param FE spaces" begin include("FEM/param_fe_spaces.jl") end
@testset "extensions" begin include("FEM/extensions.jl") end
@testset "TProduct" begin include("TProduct/tproduct.jl") end
@testset "rb steady algorithms" begin include("RB/rb_steady.jl") end
@testset "rb transient algorithms" begin include("RB/rb_transient.jl") end

@testset "snapshots" begin include("snapshots.jl") end

@testset "poisson" begin include("RBSteady/poisson.jl") end
@testset "steady stokes" begin include("RBSteady/stokes.jl") end
@testset "steady navier-stokes" begin include("RBSteady/navier_stokes.jl") end

@testset "heat equation" begin include("RBTransient/heat_equation.jl") end
@testset "unsteady elasticity" begin include("RBTransient/elasticity.jl") end
@testset "unsteady stokes" begin include("RBTransient/stokes.jl") end
@testset "unsteady navier-stokes" begin include("RBTransient/navier_stokes.jl") end
@testset "save operator" begin include("RBTransient/save_operator.jl") end

@testset "moving poisson" begin include("RBMovingGeometries/moving_poisson.jl") end
@testset "moving elasticity" begin include("RBMovingGeometries/moving_elasticity.jl") end
@testset "moving stokes" begin include("RBMovingGeometries/moving_stokes.jl") end

@testset "NeuralOperatorROMs" begin include("NeuralOperatorROMs/runtests.jl") end

end # module
