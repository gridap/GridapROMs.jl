module RBSteady

using LinearAlgebra
using LowRankApprox
using BlockArrays
using SparseArrays
using DrWatson
using Serialization

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ReferenceFEs
using Gridap.Arrays
using Gridap.Geometry
using Gridap.Fields
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs
using Gridap.TensorValues
using Gridap.Helpers

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.SolverInterfaces

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces
using ROManifolds.ParamSteady
using ROManifolds.ParamODEs
using ROManifolds.Extensions

import Base: +,-,*,\
import PartitionedArrays: tuple_of_arrays
import ROManifolds.TProduct: get_factor

export ReductionStyle
export SearchSVDRank
export FixedSVDRank
export LRApproxRank
export TTSVDRanks
export NormStyle
export EuclideanNorm
export EnergyNorm
export Reduction
export DirectReduction
export GreedyReduction
export AffineReduction
export PODReduction
export TTSVDReduction
export SupremizerReduction
export AbstractMDEIMReduction
export MDEIMReduction
export AdaptiveReduction
export get_reduction
include("ReductionMethods.jl")

export galerkin_projection
include("GalerkinProjections.jl")

export RBParamVector
include("RBParamVectors.jl")

export HRParamArray
include("HRParamArrays.jl")

export AbstractTTCore
export DofMapCore
export SparseCore
export SparseCoreCSC
include("TTCores.jl")

export contraction
export unbalanced_contractions
export sequential_product
export cores2basis
include("TTLinearAlgebra.jl")

export RBSolver
export get_fe_solver
export solution_snapshots
export residual_snapshots
export jacobian_snapshots
include("RBSolvers.jl")

export reduction
export tpod
export ttsvd
export gram_schmidt
export orth_complement!
export orth_projection
include("BasesConstruction.jl")

export Projection
export PODProjection
export TTSVDProjection
export NormedProjection
export BlockProjection
export ReducedProjection
export projection
export get_basis
export num_fe_dofs
export num_reduced_dofs
export get_cores
export project
export project!
export inv_project
export inv_project!
export union_bases
export get_norm_matrix
export enrich!
include("Projections.jl")

export RBSpace
export SingleFieldRBSpace
export MultiFieldRBSpace
export EvalRBSpace
export reduced_subspace
export reduced_spaces
export reduced_basis
export get_reduced_subspace
include("RBSpaces.jl")

export IntegrationDomain
export VectorDomain
export MatrixDomain
export vector_domain
export matrix_domain
export empirical_interpolation
export get_integration_cells
export get_cellids_rows
export get_cellids_cols
export get_owned_icells
include("IntegrationDomains.jl")

export HyperReduction
export TrivialHyperReduction
export MDEIM
export AffineContribution
export BlockHyperReduction
export get_integration_domain
export reduced_triangulation
export reduced_jacobian
export reduced_residual
export reduced_weak_form
export allocate_hypred_cache
include("HyperReductions.jl")

export BlockReindex
export collect_cell_hr_matrix
export collect_cell_hr_vector
export assemble_hr_matrix_add!
export assemble_hr_vector_add!
include("HRAssemblers.jl")

export RBOperator
export GenericRBOperator
export LinearNonlinearRBOperator
export reduced_operator
include("ReducedOperators.jl")

export ROMPerformance
export eval_performance
export create_dir
export load_snapshots
export load_residuals
export load_jacobians
export load_contribution
export load_operator
export load_results
export plot_a_solution
include("PostProcess.jl")

include("Extensions.jl")

end # module
