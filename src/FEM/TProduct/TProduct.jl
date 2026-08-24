"""
    module TProduct

Infrastructure for tensor product finite element spaces and operators on
Cartesian meshes. The key idea is that a D-dimensional Cartesian domain
``[a_1,b_1] \\times \\cdots \\times [a_D,b_D]`` can be discretised as the
tensor product of D 1D meshes, allowing bilinear forms to be assembled as
rank tensors of 1D matrices rather than full D-dimensional sparse matrices.

## FE spaces

- [`TProductFESpace`](@ref): wraps a (dof-reindexed) `space` and a vector of
  `D` 1D (dof-reindexed) `spaces_1d`, all plain Gridap `FESpace`s.

## Rank tensors

- [`Rank1Tensor`](@ref): ``a_1 \\otimes \\cdots \\otimes a_D``.
- [`GenericRankTensor`](@ref): ``\\sum_{k=1}^K a_1^k \\otimes \\cdots \\otimes a_D^k``.
- [`BlockRankTensor`](@ref): multi-field variant.

1D matrices are assembled directly on `spaces_1d` with Gridap's own
`assemble_matrix`, then packed into an `AbstractRankTensor` (`Rank1Tensor`/
`GenericRankTensor`) directly — no dedicated tensor-product assembler is
needed.
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
include("TProductGeometry.jl")

export TProductFESpace
export LexicographicFESpace
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

end # module
