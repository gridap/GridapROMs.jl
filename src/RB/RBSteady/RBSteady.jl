module RBSteady

using BlockArrays
using Clustering
using DrWatson
using LinearAlgebra
using LowRankApprox
using RadialBasisFunctions
using Random
using Serialization
using SparseArrays

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

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Extensions
using GridapROMs.Uncommon

import Base: +,-,*,\
import FillArrays: Fill
import GridapROMs.TProduct: get_factor
import PartitionedArrays: tuple_of_arrays
import Statistics: mean

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
export LocalReduction
export SupremizerReduction
export HyperReduction
export MDEIMHyperReduction
export RBFHyperReduction
export LocalHyperReduction
export AdaptiveReduction
export get_reduction
include("ReductionMethods.jl")

export galerkin_projection
include("GalerkinProjections.jl")

export RBVector
export RBParamVector
export reduced_vector
include("RBParamVectors.jl")

export HRArray
export HRParamArray
export hr_array
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
export reduced_subspace
export reduced_spaces
export reduced_basis
export get_reduced_subspace
include("RBSpaces.jl")

export IntegrationDomain
export GenericDomain
export empirical_interpolation
export get_integration_cells
export get_cell_idofs
export get_owned_icells
export move_integration_domain
include("IntegrationDomains.jl")

export Interpolation
export MDEIMInterpolation
export RBFInterpolation
export BlockInterpolation
export move_interpolation
include("Interpolations.jl")

export HRProjection
export GenericHRProjection
export MDEIMProjection
export RBFProjection
export AffineContribution
export MDEIMContribution
export RBFContribution
export BlockHRProjection
export get_interpolation
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
export assemble_hr_array_add!
include("HRAssemblers.jl")

export LocalProjection
export compute_ncentroids
export cluster
export get_local
export local_values
export get_clusters
include("LocalProjections.jl")

export RBOperator
export GenericRBOperator
export LinearNonlinearRBOperator
export AbstractLocalRBOperator
export LocalRBOperator
export reduced_operator
export change_operator
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
