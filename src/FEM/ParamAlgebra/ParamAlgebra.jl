"""
    module ParamAlgebra

Algebraic infrastructure for parametric operators and solvers.

- [`NonlinearParamOperator`](@ref) — abstract parametric nonlinear operator;
  residual and Jacobian are assembled for all parameter values simultaneously.
- [`GenericParamNonlinearOperator`](@ref) — generic concrete implementation.
- [`LazyParamNonlinearOperator`](@ref) — lazily-evaluated variant.
- [`LinearNonlinearParamOperator`](@ref) — operator split into a
  parameter-independent linear part and a parameter-dependent nonlinear part,
  enabling efficient reuse of the linear factorisation.
- [`ParamCache`](@ref) / [`SystemCache`](@ref) — caches for operator
  evaluations (residuals and Jacobians).
- Solver extensions: `solve!` overloads for `LinearSolver` and `NewtonSolver`
  acting on parametric operators.
"""
module ParamAlgebra

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers

using GridapROMs.ParamDataStructures

import ArraysOfArrays: innersize
import Gridap.ODEs: jacobian_add!
import GridapROMs.DofMaps: OIdsToIds, add_ordered_entries!
import UnPack: @unpack

include("ParamAlgebraInterfaces.jl")

export NonlinearParamOperator
export GenericParamNonlinearOperator
export LazyParamNonlinearOperator
export AbstractParamCache
export ParamCache
export SystemCache
export allocate_paramcache
export update_paramcache!
export allocate_systemcache
export update_systemcache!
include("NonlinearParamOperators.jl")

export LinearNonlinearParamOperator
export LinNonlinParamOperator
export get_linear_operator
export get_nonlinear_operator
export get_linear_systemcache
export compatible_cache
include("LinearNonlinearParamOperators.jl")

include("ParamSolvers.jl")

end # module
