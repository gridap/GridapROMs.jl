module DofMaps

using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using GridapEmbedded
using GridapEmbedded.Interfaces

using GridapROMs.Utils

import FillArrays: Fill
import Gridap.MultiField: MultiFieldFEFunction,restrict_to_field,_sum_if_first_positive
import PartitionedArrays: tuple_of_arrays
import SparseArrays: AbstractSparseMatrix
import SparseMatricesCSR: SparseMatrixCSR

export recast_indices
export recast_split_indices
export sparsify_indices
export slow_index
export fast_index
export recast
include("IndexOperations.jl")

export Range2D
export Range1D
export range_1d
export range_2d
include("Ranges.jl")

export SparsityPattern
export SparsityCSC
export TProductSparsity
export get_sparsity
export get_dof_eltype
include("SparsityPatterns.jl")

export AbstractDofMap
export TrivialDofMap
export InverseDofMap
export VectorDofMap
export TrivialSparseMatrixDofMap
export SparseMatrixDofMap
export invert
export vectorize
export flatten
export change_dof_map
include("DofMapsInterface.jl")

export get_dof_map
export get_internal_dof_map
export get_sparse_dof_map
export get_polynomial_order
export get_polynomial_orders
include("DofMapsBuilders.jl")

export DofMapArray
include("DofMapArrays.jl")

export OIdsToIds
export DofsToODofs
export OReindex
export add_ordered_entries!
include("OrderingMaps.jl")

export OrderedFESpace
export CartesianFESpace
export OrderedFEFunction
include("OrderedFESpaces.jl")

end # module
