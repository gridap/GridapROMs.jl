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

import ArraysOfArrays: _ncolons
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

export ExtensionOperator
export ExtensionLinearOperator
include("ExtensionOperators.jl")

export ExtensionParamOperator
export ExtensionLinearParamOperator
export ExtensionLinearNonlinearParamOperator
export TransientExtensionParamOperator
export TransientExtensionLinearParamOperator
export TransientExtensionLinearNonlinearParamOperator
include("ExtensionParamOperators.jl")

export ExtensionStyle
export ZeroExtension
export FunctionExtension
export HarmonicExtension
export BlockExtension
export ExtensionSolver
export extend_solution
include("ExtensionSolvers.jl")

end # module
