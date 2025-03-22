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

using ROManifolds.Utils
using ROManifolds.DofMaps
using ROManifolds.TProduct
using ROManifolds.ParamDataStructures
using ROManifolds.ParamAlgebra
using ROManifolds.ParamFESpaces
using ROManifolds.ParamSteady
using ROManifolds.ParamODEs

using ROManifolds.RBSteady

import Base: +,-,*,\
import StatsBase: countmap
import UnPack: @unpack
import ROManifolds.RBSteady: reduced_cells,_get_label

export TransientReduction
export TransientMDEIMReduction
include("ReductionMethods.jl")

include("RBSolvers.jl")

include("TTLinearAlgebra.jl")

include("GalerkinProjections.jl")

include("BasesConstruction.jl")

export TransientProjection
include("Projections.jl")

include("RBSpaces.jl")

export TransientIntegrationDomain
include("IntegrationDomains.jl")

include("HyperReductions.jl")

include("HRAssemblers.jl")

export TransientRBOperator
include("ReducedOperators.jl")

include("PostProcess.jl")

end # module
