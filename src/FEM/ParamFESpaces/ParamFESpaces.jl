"""
    module ParamFESpaces

Parametric FE spaces and FE functions for multi-sample parameter studies.

Wraps Gridap's `FESpace` layer so that every DOF array can hold a full batch of
parameter samples simultaneously.  The module provides:

- **`SingleFieldParamFESpace`** — a single-field FE space whose free-value and
  Dirichlet-value arrays are `ParamArray`s (one block per parameter sample).
- **`MultiFieldParamFESpace`** — multi-field counterpart, holding a vector of
  `SingleFieldParamFESpace`s and forwarding the Gridap `MultiFieldFESpace` interface.
- **`TrivialParamFESpace`** — wraps a plain Gridap `FESpace` as a trivially
  parametric space (same DOF values for every parameter sample).
- **`TrialParamFESpace` / `HomogeneousTrialParamFESpace`** — trial spaces with
  parameter-dependent Dirichlet boundary conditions; `TrialParamFESpace!` mutates
  an existing space in-place.
- **`UnEvalTrialFESpace` / `ParamTrialFESpace`** — lazy (unevaluated) trial FE
  spaces that instantiate Dirichlet values only when a `Realisation` is supplied.
- **`ParamFEFunction`** (`SingleFieldParamFEFunction`, `MultiFieldParamFEFunction`) —
  Gridap `FEFunction` carrying a `ParamArray` of free DOF values.
- **Assemblers** — `ParamAssemblers.jl` extends Gridap's `SparseMatrixAssembler`
  to fill parametric matrices and vectors in a single assembly pass.

The `AgFEM` tools from `GridapEmbedded` are available for aggregated FE spaces on
cut meshes; see `Extensions` for the full embedded FEM machinery.
"""
module ParamFESpaces

using LinearAlgebra
using BlockArrays
using FillArrays
using SparseArrays

using Gridap
using Gridap.Arrays
using Gridap.Algebra
using Gridap.CellData
using Gridap.FESpaces
using Gridap.Fields
using Gridap.Geometry
using Gridap.MultiField
using Gridap.ODEs
using Gridap.ReferenceFEs
using Gridap.TensorValues
using Gridap.Helpers

using GridapEmbedded
using GridapEmbedded.AgFEM

using GridapROMs.DofMaps
using GridapROMs.ParamGeometry
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.TProduct

import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction

export SingleFieldParamFESpace
export get_dirichlet_cells
export param_zero_free_values
export param_zero_dirichlet_values
include("ParamFESpaceInterface.jl")

export MultiFieldParamFESpace
include("MultiFieldParamFESpaces.jl")

export TrivialParamFESpace
include("TrivialParamFESpaces.jl")

export TrialParamFESpace
export TrialParamFESpace!
export HomogeneousTrialParamFESpace
include("TrialParamFESpaces.jl")

export UnEvalTrialFESpace
export ParamTrialFESpace
export AbstractTrialFESpace
include("UnEvalTrialFESpaces.jl")

export ParamFEFunction
export SingleFieldParamFEFunction
export MultiFieldParamFEFunction
include("ParamFEFunctions.jl")

include("ParamAssemblers.jl")

end # module
