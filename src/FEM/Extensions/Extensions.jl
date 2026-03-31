"""
    module Extensions

Extension machinery for embedded / unfitted FE methods and direct-sum spaces.

Provides the infrastructure needed when the computational mesh does not conform
to the boundary or interface of the physical domain (i.e., cut-cell / level-set
methods).  Key components:

- **`PosZeroNegReindex`** — re-index an array of DOF values according to a
  three-way split (positive / zero / negative cells), as used by `AgFEM` aggregation
  maps.  Parametric variant (`PosZeroNegParamReindex`) handles batched reindexing.
- **`MissingDofsFESpace`** — wraps an FE space and marks DOFs that are geometrically
  "missing" (outside the physical domain) so they can be filled by an extension.
- **`EmbeddedFESpace`** — an FE space on an embedded (cut) mesh, holding references
  to the background space, active space, and aggregated space together with the
  corresponding dof-id maps.
- **`DirectSumFESpace` (`⊕`)** — the direct sum of two FE spaces, used to combine,
  e.g., an interior space with an extension space.
- **`ExtensionAssembler`** — a `SparseMatrixAssembler` that assembles into the
  combined direct-sum DOF numbering.
- **`BlockExtensionSparseMatrixAssembler`** — multi-field block assembler for
  extension problems.
- **Extension solvers** — `ExtensionStyle` hierarchy (`ZeroExtension`,
  `MassExtension`, `HarmonicExtension`, `BlockExtension`) selects how missing DOF
  values are extrapolated from active DOF values.  `ExtensionSolver` and
  `ExtensionODESolver` drive steady and transient extension problems, and
  `extend_solution` is the high-level entry point.

Depends on `ParamSteady` and `ParamODEs` for operator/solver abstractions, and on
`TProduct`, `DofMaps`, `ParamFESpaces` for the underlying FE infrastructure.
"""
module Extensions

using BlockArrays
using FillArrays
using LinearAlgebra
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.FESpaces
using Gridap.Helpers
using Gridap.MultiField
using Gridap.ODEs
using Gridap.ReferenceFEs

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs

import Gridap.FESpaces: LinearConstraintsMap
import Gridap.MultiField: BlockSparseMatrixAssembler

export PosZeroNegReindex
include("PosZeroNegReindex.jl")

export MissingDofsFESpace
include("MissingDofsFESpaces.jl")

export EmbeddedFESpace
export get_bg_cell_dof_ids
export get_emb_space
export get_act_space
export get_bg_space
export complementary_space
include("EmbeddedFESpaces.jl")

export DirectSumFESpace
export ⊕
include("DirectSumFESpaces.jl")

export ExtensionAssembler
include("ExtensionAssemblers.jl")

export BlockMultiFieldExtensionStyle 
export BlockExtensionSparseMatrixAssembler
include("MultiFieldFESpaces.jl")

export ExtensionStyle
export ZeroExtension
export MassExtension
export HarmonicExtension
export BlockExtension
export ExtensionSolver
export ExtensionODESolver
export extend_solution
include("ExtensionSolvers.jl")

end # module
