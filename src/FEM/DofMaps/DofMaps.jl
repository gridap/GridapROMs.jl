module DofMaps

using BlockArrays
using DataStructures
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
export sparsify_split_indices
export slow_index
export fast_index
export recast
export group_labels
export group_ilabels
export get_group_to_labels
export get_group_to_ilabels
export inverse_table
export common_table
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
export get_common_sparsity
export get_dof_eltype
include("SparsityPatterns.jl")

export AbstractDofMap
export AbstractSparseDofMap
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
export get_sparse_dof_map
export get_cell_to_bg_cell
export get_bg_cell_to_cell
export get_bg_dof_to_dof
export get_dof_to_bg_dof
export get_bg_dof_to_active_dof
export get_active_dof_to_bg_dof
export get_bg_fdof_to_fdof
export get_fdof_to_bg_fdof
export get_bg_ddof_to_ddof
export get_ddof_to_bg_ddof
export get_dof_to_cells
include("DofMapsBuilders.jl")

export DofMapArray
include("DofMapArrays.jl")

export OIdsToIds
export DofsToODofs
export OReindex
export OTable
export add_ordered_entries!
include("OrderingMaps.jl")

export OrderedFESpace
include("OrderedFESpaces.jl")

end # module
