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

using GridapEmbedded.Interfaces

import FillArrays: Fill
import Statistics: mean

export PerformanceTracker
export CostTracker
export Speedup
export reset_tracker!
export update_tracker!
export compute_speedup
export compute_error
export compute_relative_error
export induced_norm
export sqrtreal
include("PerformanceTrackers.jl")

export PartialDerivative
export ∂₁, ∂₂, ∂₃
include("PartialDerivatives.jl")

export get_parent
export order_domains
export change_triangulation
include("Triangulations.jl")

export Contribution
export ArrayContribution
export VectorContribution
export MatrixContribution
export TupOfArrayContribution
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

end # module
