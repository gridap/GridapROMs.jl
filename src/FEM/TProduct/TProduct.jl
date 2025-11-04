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
