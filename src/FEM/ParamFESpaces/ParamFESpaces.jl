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

import Test: @test
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
