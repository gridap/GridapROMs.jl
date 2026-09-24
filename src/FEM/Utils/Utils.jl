"""
    module Utils

Foundational utilities for GridapROMs, providing:

- **Performance tracking** — [`CostTracker`](@ref), [`Speedup`](@ref),
  [`compute_error`](@ref), [`compute_relative_error`](@ref).
- **Partial derivatives** — [`PartialDerivative`](@ref), [`∂₁`](@ref),
  [`∂₂`](@ref), [`∂₃`](@ref).
- **Triangulation helpers** — [`order_domains`](@ref), [`change_triangulation`](@ref).
- **Contribution types** — [`ArrayContribution`](@ref),
  [`VectorContribution`](@ref), [`MatrixContribution`](@ref).
- **FE domain metadata** — [`FEDomains`](@ref), [`OperatorType`](@ref),
  [`JointDomains`](@ref), [`SplitDomains`](@ref),
  [`get_polynomial_order`](@ref).
"""
module Utils

using LinearAlgebra
using BlockArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ODEs
using Gridap.ReferenceFEs
using Gridap.TensorValues

import FillArrays: Fill
import Statistics: mean

export PerformanceTracker
export CostTracker
export OfflineCostTracker
export RBPerformanceTracker
export Speedup
export reset_tracker!
export update_tracker!
export set_fom_tracker!
export set_rom_tracker!
export set_subspace_tracker!
export set_jacobian_tracker!
export set_residual_tracker!
export compute_speedup
export compute_error
export compute_relative_error
export induced_norm
export sqrtabs
export get_name
include("PerformanceTrackers.jl")

export unwrap_and_setsize!
include("Unwrap.jl")

export PartialDerivative
export ∂₁, ∂₂, ∂₃
include("PartialDerivatives.jl")

export ChildTriangulation
export ChildCellQuadrature
export ChildMeasure
export order_domains
export change_triangulation
include("ChildTriangulations.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export ContributionTuple
export ArrayContributionTuple
export contribution
export get_contributions
export change_domains
export set_domains
include("Contributions.jl")

export FEDomains
export OperatorType
export LinearEq
export NonlinearEq
export LinearNonlinearEq
export TriangulationStyle
export JointDomains
export SplitDomains
export get_domains_res
export get_domains_jac
export get_polynomial_order
export get_polynomial_orders
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian
include("FEDomains.jl")

include("GridapFixes.jl")

end
