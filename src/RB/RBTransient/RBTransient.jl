module RBTransient

using LinearAlgebra
using BlockArrays
using SparseArrays
using DrWatson
using Serialization

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
using Gridap.TensorValues
using Gridap.Helpers

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Extensions

using GridapROMs.RBSteady

import Base: +,-,*,\
import StatsBase: countmap
import UnPack: @unpack
import GridapROMs.RBSteady: reduced_cells,_get_label

export HighOrderReduction
export TransientReduction
export HighOrderKroneckerReduction
export HighOrderSequentialReduction
export HighOrderHyperReduction
export HighOrderMDEIMHyperReduction
include("ReductionMethods.jl")

include("RBSolvers.jl")

include("TTLinearAlgebra.jl")

include("GalerkinProjections.jl")

export tucker
include("BasesConstruction.jl")

export TransientProjection
include("Projections.jl")

include("RBSpaces.jl")

export TransientIntegrationDomain
include("IntegrationDomains.jl")

export TransientMDEIMInterpolation
export TransientRBFInterpolation
export TransientBlockInterpolation
include("Interpolations.jl")

include("HyperReductions.jl")

include("HRAssemblers.jl")

export TransientRBOperator
include("ReducedOperators.jl")

include("PostProcess.jl")

include("Extensions.jl")

end # module
