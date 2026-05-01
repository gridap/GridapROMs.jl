"""
    module RBSteady

Reduced-basis infrastructure for steady parametric PDEs.

Implements the offline–online decomposition for parameter-dependent FE problems.
The offline stage compresses high-fidelity snapshots into a low-dimensional basis
and constructs hyper-reduced operators; the online stage solves the resulting
small system for new parameter values.  Main building blocks:

- **Reduction methods** (`ReductionMethods.jl`) — `PODReduction` (SVD-based),
  `TTSVDReduction` (tensor-train SVD), `GreedyReduction`, `AffineReduction`,
  `SupremizerReduction`, `MDEIMHyperReduction`, `SOPTHyperReduction`,
  `RBFHyperReduction`, and composites.  Rank/tolerance criteria are expressed via
  `SearchSVDRank`, `FixedSVDRank`, `LRApproxRank`, `TTSVDRanks`.

- **Bases construction** (`BasesConstruction.jl`) — `tpod` (truncated POD),
  `ttsvd` (tensor-train SVD), `gram_schmidt` / `orth_complement!`, `orth_projection`.

- **Projections** (`Projections.jl`) — `PODProjection`, `TTSVDProjection`,
  `NormedProjection` (energy norm), `BlockProjection`, `ReducedProjection`.
  Core operations: `project`/`project!`, `inv_project`/`inv_project!`,
  `union_bases`, `enrich!`.

- **RB spaces** (`RBSpaces.jl`) — `SingleFieldRBSpace`, `MultiFieldRBSpace`,
  `reduced_subspace`, `reduced_basis`.

- **Hyper-reduction** (`HyperReductions.jl`) — `HRProjection` hierarchy
  (`MDEIMProjection`, `RBFProjection`, `BlockHRProjection`) together with
  `IntegrationDomain` (DEIM-style reduced integration), `Interpolation`
  (`GreedyInterpolation`, `RBFInterpolation`), and `reduced_triangulation` /
  `reduced_jacobian` / `reduced_residual` / `reduced_weak_form`.

- **Reduced operators** (`ReducedOperators.jl`) — `GenericRBOperator`,
  `LinearNonlinearRBOperator`, `LocalRBOperator`; `reduced_operator`.

- **RB solvers** (`RBSolvers.jl`) — `RBSolver` orchestrates snapshot collection,
  offline reduction, and online solve; `solution_snapshots`, `residual_snapshots`,
  `jacobian_snapshots`.

- **Tensor-train linear algebra** (`TTLinearAlgebra.jl`) — `contraction`,
  `unbalanced_contractions`, `sequential_product`, `cores2basis`, `TTCores`.

- **Local / cluster-based projections** (`LocalProjections.jl`) — `LocalProjection`,
  `cluster`, `get_clusters`, `get_local`, `local_vals`.

- **Post-processing** (`PostProcess.jl`) — `ROMPerformance`, `eval_performance`,
  I/O helpers (`load_snapshots`, `load_contribution`, `load_operator`, …).

The `RBTransient` module extends all of the above to the time-dependent setting.
"""
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

import ArraysOfArrays: _ncolons
import Base: +,-,*,\
import FillArrays: Fill
import GridapROMs.TProduct: get_factor
import LowRankApprox: getcols,qr!,svd!,psvdrank
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
export SOPTHyperReduction
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
export DiagnosticsContribution
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
export symcholesky
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
export projection_eltype
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
export VectorDomain
export MatrixDomain
export empirical_interpolation
export s_opt
export get_integration_cells
export get_cell_irows
export get_cell_icols
export get_owned_icells
export get_interpolation_rows
export get_interpolation_cols
export move_integration_domain
include("IntegrationDomains.jl")

export Interpolation
export GreedyInterpolation
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
export allocate_hrtrian_cache
include("HyperReductions.jl")

export FetchBlockMap
export collect_cell_hr_matrix
export collect_cell_hr_vector
export assemble_hr_vector_add!
export assemble_hr_matrix_add!
include("HRAssemblers.jl")

export LocalProjection
export compute_ncentroids
export cluster
export get_local
export local_vals
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
export rom_diagnostics
export RBDiagnostics
export projection_error
export projection_diagnostics
export hr_diagnostics
export hr_error
export save_residuals
export save_jacobians
export load_residuals
export load_jacobians
export load_snapshots
export load_contribution
export load_operator
export load_results
export load_stats
export load_problem_snapshots
export try_loading_reduced_operator
export create_dir
export snapshots_label
export residuals_label
export jacobians_label
export rhs_label
export lhs_label
export test_label
export trial_label
export statistics_label
export results_label
export projection_label
export contributions_label
export linear_label
export nonlinear_label
export plot_a_solution
export plot_solutions
include("PostProcess.jl")

include("Extensions.jl")

end # module
