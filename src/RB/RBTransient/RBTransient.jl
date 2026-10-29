"""
    module RBTransient

Extension of `RBSteady` to transient parametric PDEs. Snapshots span space and
time, and so do reduced bases and hyper-reductions. Main building blocks:

- **Reduction methods** (`ReductionMethods.jl`) — [`TransientReduction`](@ref), which builds a
  [`KroneckerReduction`](@ref) (space-time POD, a Kronecker product of spatial and temporal
  bases), a [`SequentialReduction`](@ref) (TT-SVD) or a [`SteadyReduction`](@ref) (spatial
  reduction only, with time marching online); the corresponding
  [`TransientHyperReduction`](@ref)s.
- **Projections** (`Projections.jl`, `BasesConstruction.jl`) — [`TransientProjection`](@ref),
  [`KroneckerProjection`](@ref), [`SequentialProjection`](@ref).
- **Hyper-reduction** (`HyperReductions.jl`, `Interpolations.jl`) — space-time interpolation
  ([`TransientInterpolation`](@ref)), with time combinations of the time-marching scheme.
- **Reduced operators and solvers** (`RBOperators.jl`, `RBSolvers.jl`, `SpaceTime.jl`,
  `ParamTimeMarching.jl`) — [`TransientRBOperator`](@ref), solved as a single space-time
  system ([`SpaceTimeSolver`](@ref)), or by time marching for steady reductions.
- **Post-processing and I/O** (`PostProcess.jl`, `Diagnostics.jl`).
"""
module RBTransient

using BlockArrays
using Clustering
using DrWatson
using LinearAlgebra
using SparseArrays
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

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Extensions

using GridapROMs.RBSteady

import Base: +,-,*,\
import DrWatson: save
import FillArrays: Fill
import UnPack: @unpack
import Gridap.ReferenceFEs: get_order
import GridapROMs.ParamDataStructures: GenericTransientRealisation, TransientRealisationAt
import GridapROMs.RBSteady: num_centroids,get_lhs,get_rhs,_get_label,_setup,_convert_to_block,fe_dof_ids,reduced_dof_ids
import RadialBasisFunctions: Interpolator, AbstractRadialBasis, PHS

export SteadyReduction
export TransientReduction
export KroneckerReduction
export SequentialReduction
export TransientHyperReduction
export TransientTrivialHyperReduction
export TransientNoHyperReduction
export TransientAffineHyperReduction
export TransientDEIMHyperReduction
export TransientSOPTHyperReduction
export TransientRBFHyperReduction
export TransientLocalHyperReduction
export get_reduction
export get_time_combination
include("ReductionMethods.jl")

export ODERBParamVector
include("RBParamVectors.jl")

include("RBSolvers.jl")

include("TTLinearAlgebra.jl")

include("GalerkinProjections.jl")

export tucker
include("BasesConstruction.jl")

export TransientProjection
export KroneckerProjection
export SequentialProjection
export projection
export get_basis_space
export get_basis_time
export get_projection_space
export get_projection_time
export num_space_dofs
export num_times
export time_enrichment
include("Projections.jl")

include("RBSpaces.jl")

export TransientInterpolation
export InterpolationStyle
export KroneckerStyle
export SequentialStyle
export get_interpolation_style
export get_indices_time
export TransientRBFInterpolation
include("Interpolations.jl")

export TransientAffineHRContribution
export TransientAffineHRContributionTuple
export get_common_time_domain
include("HyperReductions.jl")

include("HRAssemblers.jl")

include("LocalProjections.jl")

include("LocalHyperReductions.jl")

export TransientROMOperator
export TransientRBOperator
include("RBOperators.jl")

export SpaceTimeParamOperator
export SpaceTimeSolver
include("SpaceTime.jl")

include("ParamTimeMarching.jl")

include("PostProcess.jl")

end
