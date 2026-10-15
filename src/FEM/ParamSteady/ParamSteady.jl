"""
    module ParamSteady

Parametric FE operators and solvers for steady parametric PDEs.

Extends Gridap's `FEOperator`/`FESpaces` layer with parameter awareness, so that
residuals and Jacobians can be evaluated at a `Realisation` (a batch of parameter
samples).  The module introduces:

- **Operator types** — `LinearParamEq`, `NonlinearParamEq`, 
  tag whether a parametric operator is linear, nonlinear, or a split linear + nonlinear
  combination.
- **Low-level algebraic operators** — `ParamOperator`, `LinearParamOperator`, wrap
  assembled matrices/vectors and expose the `residual!`/`jacobian!` interface at
  parameter level.
- **FE-level operators** — `ParamFEOperator`, `LinearParamFEOperator`
  manage Gridap assemblers, FE domains, and parameter-to-sample dispatch. A linear +
  nonlinear split operator is just a `ParamFEOperator`
  whose `res`/`jac`/`domains` fields are each a `Bundle` of the linear and nonlinear
  halves (`T=BundleForm`); `get_linear_part`/`get_nonlinear_part` split it back apart.
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
export ParamOperator
export PlainParamOperator
export BundleParamOperator
export LinearParamOperator
export get_fe_operator
export get_jac
include("ParamOperators.jl")

export ParamFEOperator
export PlainParamFEOperator
export BundleParamFEOperator
export LinearParamFEOperator
export GenericParamOperator
export get_param_space
export get_param_assembler
include("ParamFEOperators.jl")

include("ParamFESolvers.jl")

end
