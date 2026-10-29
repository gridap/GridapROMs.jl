"""
    module RBSteady

Reduced-basis infrastructure for steady parametric PDEs: offline reduction and
hyper-reduction, and online solution of the reduced problems. Main building blocks:

- **Reduction methods** (`ReductionMethods.jl`) — [`Reduction`](@ref), [`PODReduction`](@ref),
  [`TTSVDReduction`](@ref), [`LocalReduction`](@ref), [`SupremizerReduction`](@ref), and the
  hyper-reductions [`DEIMHyperReduction`](@ref), [`SOPTHyperReduction`](@ref),
  [`RBFHyperReduction`](@ref), [`NoHyperReduction`](@ref), [`AffineHyperReduction`](@ref).
- **Norms and couplings** (`EnergyNorms.jl`) — [`L2`](@ref), [`H1`](@ref), [`EnergyNorm`](@ref),
  [`DivCoupling`](@ref), [`BlockNorm`](@ref).
- **Bases construction** (`BasesConstruction.jl`) — [`tpod`](@ref), [`ttsvd`](@ref),
  [`gram_schmidt`](@ref), [`orth_projection`](@ref).
- **Projections and RB spaces** (`Projections.jl`, `RBSpaces.jl`) — [`Projection`](@ref),
  [`RBSpace`](@ref), [`reduced_spaces`](@ref), [`project`](@ref), [`inv_project`](@ref),
  [`galerkin_projection`](@ref).
- **Hyper-reduction** (`HyperReductions.jl`, `IntegrationDomains.jl`, `Interpolations.jl`) —
  [`HRProjection`](@ref), [`DEIM`](@ref), [`SOPT`](@ref), [`reduced_weak_form`](@ref).
- **Local ROMs** (`LocalProjections.jl`, `LocalHyperReductions.jl`) — [`LocalProjection`](@ref),
  [`get_local`](@ref), [`compute_ncentroids`](@ref).
- **Reduced operators and solvers** (`RBOperators.jl`, `RBSolvers.jl`) — [`RBSolver`](@ref),
  [`RBOperator`](@ref), [`reduced_operator`](@ref), [`solution_snapshots`](@ref).
- **Tensor-train linear algebra** (`TTCores.jl`, `TTLinearAlgebra.jl`) — [`contraction`](@ref),
  [`sequential_product`](@ref), [`cores2basis`](@ref).
- **Post-processing and I/O** (`PostProcess.jl`, `Diagnostics.jl`) — [`rom_performance`](@ref),
  [`rom_diagnostics`](@ref), [`load_operator`](@ref), [`load_snapshots`](@ref).

The `RBTransient` module extends all of the above to the time-dependent setting.
"""
module RBSteady

using BlockArrays
using Clustering
using DrWatson
using LinearAlgebra
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

import ArraysOfArrays: _ncolons
import Base: +,-,*,\
import DrWatson: save,load
import FillArrays: Fill
import Gridap.Polynomials: Monomial
import GridapROMs.TProduct: get_factor
import PartitionedArrays: i_am_main, tuple_of_arrays
import RadialBasisFunctions: Interpolator, AbstractRadialBasis, PHS, MonomialBasis, _build_collocation_matrix!
import Statistics: mean

export ReductionStyle
export NoReductionStyle
export SearchSVDRank
export FixedSVDRank
export TTSVDRanks
export AssembleOperator
export assemble_operator
export NormStyle
export ℓ2
export EuclideanNorm
export L2
export H1
export NitscheH1
export EnergyNorm
export CouplingStyle
export DivCoupling
export BlockOperator
export BlockNorm
export BlockCoupling
export l2_norm
export h1_norm
export div_coupling
export Reduction
export NoReduction
export DirectReduction
export GreedyReduction
export PODReduction
export TTSVDReduction 
export LocalReduction
export SupremizerReduction
export HyperReduction
export TrivialHyperReduction
export NoHyperReduction
export AffineHyperReduction
export DEIMHyperReduction
export SOPTHyperReduction
export RBFHyperReduction
export LocalHyperReduction
export AdaptiveReduction
export get_reduction
include("ReductionMethods.jl")

export galerkin_projection
export galerkin_projection!
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

export ROMSolver
export RBSolver
export GlobalRBSolver
export LocalRBSolver
export get_fe_solver
export get_reduced_solver
export solution_snapshots
export update_rom_tracker!
include("RBSolvers.jl")

export reduction
export tpod
export ttsvd
export symcholesky
export gram_solver
export symmetrise!
export gram_schmidt
export orth_complement!
export orth_projection
include("BasesConstruction.jl")

export Projection
export GenericProjection
export PODProjection
export TTSVDProjection
export NormedProjection
export BlockProjection
export ReducedProjection
export ReducedVecProjection
export ReducedMatProjection
export GalerkinProjectable
export projection
export num_fe_dofs
export num_reduced_dofs
export isnull
export get_basis
export get_cores
export get_recast_basis
export get_recast_cores
export project
export project!
export inv_project
export inv_project!
export projection_eltype
export union_bases
export get_norm_matrix
export enrich!
export supremizers
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
export DEIM
export SOPT
export get_integration_cells
export get_cell_idofs
export get_interpolation_dofs
include("IntegrationDomains.jl")

export Interpolation
export EmptyInterpolation
export GreedyInterpolation
export RBFInterpolation
export BlockInterpolation
include("Interpolations.jl")

export HRProjection
export HRVecProjection
export HRMatProjection
export BlockHRProjection
export GenericHRProjection
export AffineContribution
export get_style
export get_interpolation
export reduced_jacobian
export reduced_residual
export reduced_weak_form
export residual_contribution
export jacobian_contribution
export allocate_hypred_cache
include("HyperReductions.jl")

export collect_cell_hr_matrix
export collect_cell_hr_vector
export assemble_hr_array_add!
export fetch_block
include("HRAssemblers.jl")

export LocalProjection
export compute_ncentroids
export cluster
export get_local
export local_vals
export get_clusters
include("LocalProjections.jl")

export LocalInterpolation
export LocalHRProjection
export LocalHRContribution
include("LocalHyperReductions.jl")

export ROMOperator
export RBOperator
export reduced_operator
export change_operator
include("RBOperators.jl")

export RBPerformanceTracker
export OfflineCostTracker
export rom_performance
export rom_diagnostics
export RBDiagnostics
export DiagnosticsContribution
export projection_error
export projection_diagnostics
export hr_diagnostics
export hr_error
export load_snapshots
export load_contribution
export load_projection
export load_subspace
export load_operator
export load_results
export load_stats
export load_problem_snapshots
export SNAPSHOTS_LABEL
export RESIDUALS_LABEL
export JACOBIANS_LABEL
export RHS_LABEL
export LHS_LABEL
export TEST_LABEL
export TRIAL_LABEL
export STATISTICS_LABEL
export RESULTS_LABEL
export PROJECTION_LABEL
export CONTRIBUTIONS_LABEL
export LINEAR_LABEL
export NONLINEAR_LABEL
export OFFLINE_LABEL
export ONLINE_LABEL
include("PostProcess.jl")

include("Extensions.jl")

end
