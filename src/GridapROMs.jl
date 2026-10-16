module GridapROMs

include("FEM/Utils/Utils.jl")

include("FEM/DofMaps/DofMaps.jl")

include("FEM/TProduct/TProduct.jl")

include("FEM/ParamDataStructures/ParamDataStructures.jl")

include("FEM/ParamAlgebra/ParamAlgebra.jl")

include("FEM/ParamGeometry/ParamGeometry.jl")

include("FEM/ParamFESpaces/ParamFESpaces.jl")

include("FEM/ParamSteady/ParamSteady.jl")

include("FEM/ParamODEs/ParamODEs.jl")

include("FEM/Extensions/Extensions.jl")

include("RB/RBSteady/RBSteady.jl")

include("RB/RBTransient/RBTransient.jl")

include("Distributed/Distributed.jl")

include("Exports.jl")

# Examples / visualization interface. Methods live in `ext/GridapROMsExt.jl` and
# become available once `Plots` is loaded (`using Plots`).
function try_loading_fe_snapshots end
function try_loading_online_fe_snapshots end
function try_loading_reduced_operator end
function update_reduction end
function update_solver end
function run_test end
function plot_solutions end
function plot_errors end
export try_loading_fe_snapshots, try_loading_online_fe_snapshots
export try_loading_reduced_operator, update_reduction, update_solver, run_test
export plot_solutions, plot_errors

end
