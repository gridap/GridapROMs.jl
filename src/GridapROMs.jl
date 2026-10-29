"""
    module GridapROMs

Reduced order models of parameterised PDEs, built on top of Gridap. See the
tutorials in the documentation for an introduction, and the submodules for the
full API
"""
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

# examples / visualization interface (see `ext/GridapROMsExt.jl`)
# they become available once `Plots` is loaded

"""
    try_loading_fe_snapshots(dir,rbsolver,feop,args...;label="",kwargs...) -> AbstractSnapshots

Loads the FE snapshots stored in `dir`, or computes and saves them if loading fails.
Requires `Plots`
"""
function try_loading_fe_snapshots end

"""
    try_loading_reduced_operator(dir,rbsolver,feop,fesnaps) -> ROMOperator

Loads the reduced operator stored in `dir`, or runs the offline phase on the
snapshots `fesnaps` and saves its output if loading fails. Requires `Plots`
"""
function try_loading_reduced_operator end

"""
    update_reduction(red::Reduction,tolrank) -> Reduction

Copy of `red` with tolerance (or rank) `tolrank`. Requires `Plots`
"""
function update_reduction end

"""
    update_solver(rbsolver::RBSolver,tolrank) -> RBSolver

Copy of `rbsolver` with state tolerance (or rank) `tolrank`, and hyper-reduction
tolerance `tolrank*1e-2` (or rank `tolrank+5`). Requires `Plots`
"""
function update_solver end

"""
    run_test(dir,rbsolver,feop,tolranks=[1e-1,1e-2,1e-3,1e-4,1e-5],args...;nparams=10,sampling=:uniform) -> Vector

Builds and tests a ROM for each tolerance (or rank) in `tolranks`, on `nparams`
online parameters. Snapshots, reduced operators and results are stored in `dir`, and
the convergence plot in `dir/results`. Requires `Plots`
"""
function run_test end

"""
    plot_solutions(dir,rbop::ROMOperator,sol::Snapshots,sol_approx::Snapshots;kwargs...)

Writes to VTK files in `dir` the FE solutions `sol`, the ROM solutions `sol_approx`,
and their difference. Requires `Plots`
"""
function plot_solutions end

"""
    plot_errors(dir,tolranks,perfs::AbstractVector{<:RBPerformanceTracker})

Plots the errors in `perfs` against the tolerances (or ranks) `tolranks`, and saves
the figure in `dir/convergence.png`. Requires `Plots`
"""
function plot_errors end

export try_loading_fe_snapshots
export try_loading_reduced_operator
export update_reduction
export update_solver
export run_test
export plot_solutions
export plot_errors

end
