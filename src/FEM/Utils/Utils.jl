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

export PerformanceTracker
export CostTracker
export Speedup
export reset_tracker!
export update_tracker!
export compute_speedup
export compute_error
export compute_relative_error
export induced_norm
include("PerformanceTrackers.jl")

export PartialDerivative
export ∂₁, ∂₂, ∂₃
include("PartialDerivatives.jl")

export get_parent
export order_domains
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
export TriangulationStyle
export JointDomains
export SplitDomains
include("FEDomains.jl")

export FEDomainOperator
export LinearFEOperator
export DomainOperator
export GenericFEDomainOperator
export GenericDomainOperator
export get_fe_operator
export get_jac
export get_domains
export get_domains_res
export get_domains_jac
export get_polynomial_order
export get_polynomial_orders
include("FEDomainOperators.jl")

export DomainOperator
export GenericDomainOperator
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian
include("DomainOperators.jl")

end # module
