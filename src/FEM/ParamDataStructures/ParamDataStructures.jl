"""
    module ParamDataStructures

Data structures for parametric PDEs: parameter spaces, realisations, parametric
arrays, sparse matrices, and solution snapshots.

## Parameter spaces and sampling

- [`ParamSpace`](@ref) / [`TransientParamSpace`](@ref) — domains over which
  parameters vary, with sampling strategies [`UniformSampling`](@ref),
  [`NormalSampling`](@ref), [`HaltonSampling`](@ref), etc.
- [`Realisation`](@ref) / [`TransientRealisation`](@ref) — concrete parameter
  (and time) values drawn from a `ParamSpace`.
- [`ParamFunction`](@ref) / [`TransientParamFunction`](@ref) — functions
  parameterised over a `ParamSpace`.

## Parametric arrays

- [`AbstractParamArray`](@ref) — abstract supertype; concrete subtypes are
  `ConsecutiveParamArray` (contiguous data layout, most efficient) and
  `GenericParamArray`.
- [`ParamSparseMatrix`](@ref) — sparse matrix with one entry per parameter.
- [`BlockParamArray`](@ref) — block-structured parametric array for
  multi-field problems.

## Snapshots

- [`Snapshots`](@ref) — collection of FE solution vectors indexed by
  parameter (and optionally time).
- [`TransientSnapshots`](@ref) and its subtypes for time-dependent problems.
- [`select_snapshots`](@ref) — extract a subset of snapshots.
"""
module ParamDataStructures

using LinearAlgebra
using BlockArrays
using ForwardDiff
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.Fields
using Gridap.Geometry
using Gridap.CellData
using Gridap.Helpers

using GridapROMs.Utils
using GridapROMs.DofMaps

import ArraysOfArrays: front_tuple,innersize,_ncolons
import Base:+,-,*,/,\,^
import Distributions: Uniform,Normal
import FillArrays: Fill
import HaltonSequences: HaltonPoint
import LinearAlgebra: ⋅
import Random: GLOBAL_RNG,randperm
import StatsBase: sample
import Gridap.Fields: BroadcastOpFieldArray,BroadcastingFieldOpMap,LinearCombinationField,LinearCombinationMap,AffineMap,ConstantMap
import Gridap.ReferenceFEs: LagrangianDofBasis
import Gridap.TensorValues: ⊗, ⊙
import SparseArrays.getcolptr

export UniformSampling
export NormalSampling
export HaltonSampling
export LatinHypercubeSampling
export TensorialUniformSampling
include("Sampling.jl")

export AbstractRealisation
export Realisation
export TransientRealisation
export ParamSpace
export TransientParamSpace
export AbstractParamFunction
export ParamFunction
export TransientParamFunction
export realisation
export get_params
export get_times
export get_at_time
export get_at_timestep
export num_params
export num_times
export get_initial_time
export get_final_time
export shift!
include("ParamSpaces.jl")

export AbstractParamData
export eltype2
export parameterise
export local_parameterise
export get_param_data
export param_length
export param_eachindex
export param_getindex
export param_setindex!
export get_param_entry
export get_param_entry!
include("ParamDataInterface.jl")

export ParamBlock
export GenericParamBlock
export TrivialParamBlock
include("ParamBlocks.jl")

export AbstractParamArray
export AbstractParamVector
export AbstractParamMatrix
export ParamArray
export ParamVector
export ParamMatrix
export GenericParamArray
export innersize
export innerlength
export inneraxes
include("ParamArraysInterface.jl")

export TrivialParamArray
export ConsecutiveParamArray
export ConsecutiveParamVector
export ConsecutiveParamMatrix
export GenericParamVector
export GenericParamMatrix
export get_all_data
export param_cat
include("ParamArrays.jl")

export ParamSparseMatrix
export ParamSparseMatrixCSC
export ParamSparseMatrixCSR
export ConsecutiveParamSparseMatrixCSC
export GenericParamSparseMatrixCSC
export ConsecutiveParamSparseMatrixCSR
export GenericParamSparseMatrixCSR
export ConsecutiveParamSparseMatrix
include("ParamSparseMatrices.jl")

export BlockParamArray
export BlockParamVector
export BlockParamMatrix
export BlockConsecutiveParamArray
export BlockConsecutiveParamVector
export BlockConsecutiveParamMatrix
include("BlockParamArrays.jl")

export ParamVectorWithEntryRemoved
export ParamVectorWithEntryInserted
include("ParamVectorWithEntries.jl")

export AbstractSnapshots
export Snapshots
export SteadySnapshots
export GenericSnapshots
export ReshapedSnapshots
export SnapshotsAtIndices
export SparseSnapshots
export BlockSnapshots
export get_realisation
export select_snapshots
export num_space_dofs
include("Snapshots.jl")

export TransientSnapshots
export TransientGenericSnapshots
export TransientSnapshotsWithIC
export TransientSparseSnapshots
export UnfoldingTransientSnapshots
export ModeTransientSnapshots
export get_initial_data
export get_initial_param_data
export get_mode1
export get_mode2
export change_mode
include("TransientSnapshots.jl")

include("ParamBroadcasts.jl")

export FetchParam
export lazy_param_getindex
export lazy_testitem
include("ParamMaps.jl")

end # module
