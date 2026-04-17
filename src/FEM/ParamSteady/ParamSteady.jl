"""
    module ParamSteady

Parametric FE operators and solvers for steady parametric PDEs.

Extends Gridap's `FEOperator`/`FESpaces` layer with parameter awareness, so that
residuals and Jacobians can be evaluated at a `Realisation` (a batch of parameter
samples).  The module introduces:

- **Operator types** — `LinearParamEq`, `NonlinearParamEq`, `LinearNonlinearParamEq`
  tag whether a parametric operator is linear, nonlinear, or a split linear + nonlinear
  combination.
- **Low-level algebraic operators** — `ParamOperator`, `JointParamOperator`,
  `SplitParamOperator`, `LinearParamOperator`, `LinearNonlinearParamOperator` wrap
  assembled matrices/vectors and expose the `residual!`/`jacobian!` interface at
  parameter level.
- **FE-level operators** — `ParamFEOperator` (joint or split), `LinearParamFEOperator`,
  `LinearNonlinearParamFEOperator` manage Gridap assemblers, FE domains, and
  parameter-to-sample dispatch.
- **Solvers** — thin wrappers in `ParamFESolvers.jl` dispatch parametric solves to
  the underlying Gridap / GridapSolvers linear or nonlinear solvers.

Downstream modules (`ParamODEs`, `RBSteady`, `Extensions`) all build on the
abstractions defined here.
"""
module ParamSteady

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Helpers

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using GridapSolvers.MultilevelTools

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces

import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order
import GridapROMs.Utils: CostTracker,change_domains,set_domains

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export ParamOperator
export JointParamOperator
export SplitParamOperator
export LinearParamOperator
export LinearNonlinearParamOperator
export get_fe_operator
export get_jac
include("ParamOperators.jl")

export ParamFEOperator
export SplitParamFEOperator
export JointParamFEOperator
export LinearParamFEOperator
export FEDomains
export get_param_space
export get_param_assembler
include("ParamFEOperators.jl")

export LinearNonlinearParamFEOperator
export join_operators
include("LinearNonlinearParamFEOperators.jl")

export GenericParamOperator
export GenericLinearNonlinearParamOperator
include("GenericParamOperators.jl")

include("ParamFESolvers.jl")

end # module
