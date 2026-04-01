"""
    module ParamODEs

Parametric FE operators and solutions for transient parametric PDEs.

Extends `ParamSteady` to the time-dependent setting, building on Gridap's `ODEs`
machinery.  Key components:

- **Time derivatives** — `∂ₚt` and `∂ₚtt`, parameter-aware analogues of Gridap's
  `∂t`/`∂tt`, for use inside weak forms.
- **Transient trial spaces** — `TransientTrialParamFESpace` and
  `TransientMultiFieldParamFESpace` add time-varying Dirichlet conditions to
  parametric FE spaces.
- **ODE operators** — `ODEParamOperator` hierarchy (`JointODEParamOperator`,
  `SplitODEParamOperator`, `LinearNonlinearODEParamOperator`, …) translate a
  transient parametric weak form into the residual/Jacobian signature expected by
  Gridap's time-marching schemes.
- **Stage operators** — `ParamStageOperator` wraps a single ODE stage solve,
  keeping track of the current time, stage coefficients, and parameter realisation.
- **Transient FE operators** — `TransientParamFEOperator` and specialisations
  (`SplitTransientParamFEOperator`, `TransientLinearParamFEOperator`, …) combine
  the spatial FE machinery with time-derivative information.
- **ODE solutions** — `ODEParamSolution` iterates over time steps, yielding
  `(realisation, FEFunction)` pairs; `TransientParamFESolution` wraps this
  iterator into a `ParamFEFunction` per time step.
- **Shifted solvers** — `ShiftedSolver` applies a constant operator shift to a
  linear solver, enabling efficient generalised-alpha or implicit-explicit schemes.

Modules `RBTransient` and `Extensions` depend on the abstractions defined here.
"""
module ParamODEs

using LinearAlgebra
using ForwardDiff

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Helpers

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady

import BlockArrays: blocks,blocklength,mortar
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import GridapROMs.Utils: change_domains,set_domains,get_domains_res,get_domains_jac

export ∂ₚt,∂ₚtt
include("TimeDerivatives.jl")

include("TransientParamCellFields.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpaces.jl")

include("TransientNonlinearParamOperators.jl")

export ODEParamOperatorType
export NonlinearParamODE
export LinearParamODE
export LinearNonlinearParamODE
export ODEParamOperator
export JointODEParamOperator
export SplitODEParamOperator
export LinearNonlinearODEParamOperator
export TransientLinearParamOperator
export TransientParamOperator
export LinearNonlinearTransientParamOperator
include("ODEParamOperators.jl")

export ParamStageOperator
include("ParamStageOperators.jl")

export TransientParamFEOperator
export SplitTransientParamFEOperator
export JointTransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientLinearParamFEOperator
export TransientLinearParamFEOpFromWeakForm
include("TransientParamFEOperators.jl")

include("ParamTimeMarching.jl")

export ShiftedSolver
include("ShiftedSolvers.jl")

export ODEParamSolution
export initial_conditions
include("ODEParamSolutions.jl")

include("TransientParamFESolutions.jl")

end # module
