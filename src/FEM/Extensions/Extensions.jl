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

export Extension
export ZeroExtension
export FunctionExtension
export HarmonicExtension
include("ExtensionInterface.jl")

export ParamExtension
include("ParamExtensions.jl")

export ExternalFESpace
export ExternalAgFEMSpace
export OrderedAgFEMSpace
include("ExternalFESpaces.jl")

export ExtensionFESpace
export ZeroExtensionFESpace
export FunctionExtensionFESpace
export HarmonicExtensionFESpace
export ParamZeroExtensionFESpace
export ParamFunctionExtensionFESpace
export ParamHarmonicExtensionFESpace
export ExtendedFEFunction
export get_bg_cell_dof_ids
export extend_free_values
export extend_dirichlet_values
export extend_free_dirichlet_values
export extended_interpolate
export extended_interpolate_everywhere
export extended_interpolate_dirichlet
include("ExtensionFESpaces.jl")

export ExtensionAssembler
export ExtensionAssemblerInsertIn
export ExtensionAssemblerInsertOut
export ExtensionAssemblerInsertInOut
export assemble_extended_vector
export assemble_extended_matrix
include("ExtensionAssemblers.jl")

export ExtensionParamOperator
include("ExtensionOperators.jl")

end # module
