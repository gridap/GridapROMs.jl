module Distributed

using DrWatson
using LinearAlgebra
using SparseArrays
using Serialization
using PartitionedArrays

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
using GridapDistributed

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.RBSteady

import ArraysOfArrays: innersize
import BlockArrays: mortar
import Gridap.Helpers: @abstractmethod, @check
import GridapDistributed: BlockPArray, DistributedFESpace, DistributedSingleFieldFESpace, DistributedMultiFieldFESpace, to_parray_of_arrays
import GridapROMs.DofMaps: range_2d, range_1d
import GridapROMs.ParamAlgebra: ParamBuilder, ParamCounter
import MPI
import PartitionedArrays: VectorAssemblyCache, length_to_ptrs!, rewind_ptrs!, getany

include("ParamJaggedArrays.jl")

include("ParamArraysInterface.jl")

include("ParamSparseUtils.jl")

include("Primitives.jl")

include("ParamAlgebra.jl")

include("ParamFESpaces.jl")

include("ParamSolvers.jl")

export GenericPArray
include("GenericPArray.jl")

export DistributedSnapshots
include("Snapshots.jl")

end
