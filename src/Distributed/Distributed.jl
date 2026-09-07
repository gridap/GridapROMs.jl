"""
    module Distributed

MPI-parallel extensions of the GridapROMs parametric and reduced-basis layer.

Wraps the distributed data structures provided by `PartitionedArrays` and
`GridapDistributed` so that parametric snapshots and FE solves can be performed
in parallel.  Key components:

- **Distributed `ParamArray`s** — `OwnAndGhostParamVector` and
  `ParamJaggedArray` carry multi-sample DOF arrays in the owned-and-ghost
  partitioning expected by `PartitionedArrays`.  `ParamArraysInterface.jl`
  implements the full `AbstractParamArray` interface for these types.

- **Sparse utilities** — `ParamSparseUtils.jl` provides distributed sparse
  matrix/vector assembly helpers (CSR row-pointer arithmetic,assembly caches)
  compatible with `ParamArray` entries.

- **Primitives** — `Primitives.jl` contains low-level MPI-aware operations
  (scatter/gather,consistent local-size queries) reused by the higher-level
  components.

- **Distributed algebra** — `ParamAlgebra.jl` extends `ParamBuilder` /
  `ParamCounter` to the distributed setting; `ParamSolvers.jl` wraps
  `GridapDistributed` solvers for parametric systems.

- **Distributed FE spaces** — `ParamFESpaces.jl` specialises
  `DistributedSingleFieldFESpace` / `DistributedMultiFieldFESpace` for
  parametric DOF arrays.

- **Distributed snapshots** — `GenericPArray` is a `PartitionedArrays`
  `PVector`-like container for generic parallel data; `DistributedSnapshots`
  wraps `Snapshots` for the distributed case.

Requires `GridapDistributed` and MPI; not loaded unless `Distributed` is
explicitly used.
"""
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

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.RBSteady
using GridapROMs.RBTransient

import ArraysOfArrays: innersize,_ncolons
import BlockArrays: BlockVector,BlockMatrix,BlockArray,mortar,blocks
import Gridap.Helpers: @abstractmethod,@check,@notimplemented,@notimplementedif
import GridapDistributed: BlockPMatrix,BlockPVector,BlockPArray,DistributedFESpace,DistributedSingleFieldFESpace,DistributedMultiFieldFESpace,DistributedTriangulation,to_parray_of_arrays
import GridapROMs.DofMaps: range_2d,range_1d
import GridapROMs.ParamAlgebra: ParamBuilder,ParamCounter
import GridapROMs.RBSteady: get_l2_form,get_h1_form,get_div_coupling_form,l2_norm,h1_norm,div_coupling,_assemble_operator,_unwrap,_meas,_energy_mortar,_coupling_mortar
import MPI
import PartitionedArrays: SubSparseMatrix,VectorAssemblyCache,length_to_ptrs!,rewind_ptrs!,getany

include("OwnAndGhostParamVectors.jl")

include("ParamJaggedArrays.jl")

include("ParamArraysInterface.jl")

include("ParamSparseUtils.jl")

include("Primitives.jl")

include("ParamAlgebra.jl")

include("ParamFESpaces.jl")

include("ParamSolvers.jl")

include("NZIndexPartitions.jl")

export GenericPArray
include("GenericPArray.jl")

export DistributedSnapshots
include("Snapshots.jl")

include("EnergyNorms.jl")

include("BasesConstruction.jl")

include("Reduced.jl")

end
