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

using GridapEmbedded
using GridapEmbedded.AgFEM

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces
using ROManifolds.ParamSteady
using ROManifolds.ParamODEs

import Gridap.FESpaces: LinearConstraintsMap

export get_bg_dof_to_dof
export get_dof_to_bg_dof
include("DofUtils.jl")

include("ODofUtils.jl")

export EmbeddedFESpace
export get_bg_cell_dof_ids
include("EmbeddedFESpaces.jl")

export DirectSumFESpace
export ⊕
include("DirectSumFESpaces.jl")

export ExtensionAssembler
include("ExtensionAssemblers.jl")

export ExtensionParamOperator
export ExtensionLinearParamOperator
export ExtensionLinearNonlinearParamOperator
include("ExtensionParamOperators.jl")

export ZeroExtension
export FunctionExtension
export HarmonicExtension
export ExtensionSolver
export extend_solution
export pad_solution!
include("ExtensionSolvers.jl")

end # module
