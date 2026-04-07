"""
    module TProduct

Infrastructure for tensor product finite element spaces and operators on
Cartesian meshes. The key idea is that a D-dimensional Cartesian domain
``[a_1,b_1] \\times \\cdots \\times [a_D,b_D]`` can be discretised as the
tensor product of D 1D meshes, allowing bilinear forms to be assembled as
rank tensors of 1D matrices rather than full D-dimensional sparse matrices.

## Geometry

- [`TProductDiscreteModel`](@ref): D-dimensional `CartesianDiscreteModel`
  together with its D 1D factor models.
- [`TProductTriangulation`](@ref): triangulation on a `TProductDiscreteModel`.
- [`TProductMeasure`](@ref): quadrature measure composed of D 1D measures.

## FE spaces

- [`TensorProductReferenceFE`](@ref): a `ReferenceFE` that carries the 1D
  factor reffes and triggers `TProductFESpace` construction via the standard
  `FESpace(model,reffe;kwargs...)` interface.
- [`TProductFESpace`](@ref): wraps an `OrderedFESpace` and D 1D
  `OrderedFESpace`s.

## Cell data

- [`TProductFEBasis`](@ref): FE basis in separated form; supports `gradient`
  and `PartialDerivative`.
- [`GenericTProductCellField`](@ref),
  [`GenericTProductDiffCellField`](@ref): cell fields in separated form.

## Rank tensors

- [`Rank1Tensor`](@ref): ``a_1 \\otimes \\cdots \\otimes a_D``.
- [`GenericRankTensor`](@ref): ``\\sum_{k=1}^K a_1^k \\otimes \\cdots \\otimes a_D^k``.
- [`BlockRankTensor`](@ref): multi-field variant.

## Assembly

- [`TProductSparseMatrixAssembler`](@ref): assembles 1D matrices and packs
  them into an [`AbstractRankTensor`](@ref).
- [`TProductBlockSparseMatrixAssembler`](@ref): multi-field variant.
"""
module TProduct

using LinearAlgebra
using BlockArrays
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using GridapEmbedded
using GridapEmbedded.Interfaces

using GridapROMs.Utils
using GridapROMs.DofMaps

import Base:+,-
import FillArrays: Fill,fill
import Gridap.ReferenceFEs: get_order

export TProductDiscreteModel
export TProductTriangulation
export TProductMeasure
include("TProductGeometry.jl")

export TProductFESpace
export get_tp_fe_basis
export get_tp_trial_fe_basis
include("TProductFESpaces.jl")

export AbstractRankTensor
export Rank1Tensor
export GenericRankTensor
export BlockRankTensor
export MatrixOrTensor
export get_factors
export get_decomposition
export get_crossnorm
include("RankTensors.jl")

include("TProductCellData.jl")

export TProductSparseMatrixAssembler
export TProductBlockSparseMatrixAssembler
include("TProductAssembly.jl")

end # module
