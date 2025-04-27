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

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs

import Gridap.FESpaces: LinearConstraintsMap

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
export TransientExtensionParamOperator
export TransientExtensionLinearParamOperator
export TransientExtensionLinearNonlinearParamOperator
include("ExtensionParamOperators.jl")

export ZeroExtension
export FunctionExtension
export HarmonicExtension
export ExtensionSolver
export extend_solution
include("ExtensionSolvers.jl")

end # module
