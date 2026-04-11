"""
    module RBTransient

Reduced-basis infrastructure for transient parametric PDEs.

Extends `RBSteady` to the space–time setting.  The additional complexity arises
from the time dimension: snapshots are matrices (space × time) and bases must
capture temporal as well as spatial structure.  Key extensions:

- **Reduction methods** — `TransientReduction`, `KroneckerReduction`,
  `SequentialReduction` extend the steady reductions; `HighDimReduction`,
  `SteadyReduction` handle the "high-dimensional" (full-space) and purely-spatial
  cases.  Corresponding hyper-reduction variants: `TransientHyperReduction`,
  `HighDimMDEIMHyperReduction`, `HighDimSOPTHyperReduction`.

- **Tucker / Kronecker bases** (`BasesConstruction.jl`) — `tucker` computes a
  Tucker decomposition, used for Kronecker-product basis representations.

- **Transient projections** (`Projections.jl`) — `TransientProjection` wraps a
  spatial projection with a temporal one.

- **Transient integration domains** (`IntegrationDomains.jl`) —
  `TransientIntegrationDomain` extends DEIM-style reduced integration to include
  a reduced time-index set.

- **Transient interpolations** (`Interpolations.jl`) —
  `TransientGreedyInterpolation`, `TransientRBFInterpolation`,
  `TransientBlockInterpolation`.

- **Transient operators** (`ReducedOperators.jl`) — `TransientRBOperator` adds
  time-stepping to the reduced operator interface.

- **Time marching** (`ParamTimeMarching.jl`) — hooks into `ParamODEs` to drive
  the online transient solve with the reduced operator.

- **Post-processing and I/O** (`PostProcess.jl`) — transient counterpart of the
  steady post-processing utilities.

All space-only functionality (RB spaces, hyper-reduction infrastructure, linear
algebra) is imported directly from `RBSteady`.
"""
module RBTransient

using BlockArrays
using Clustering
using DrWatson
using LinearAlgebra
using RadialBasisFunctions
using SparseArrays
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

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using GridapSolvers.SolverInterfaces

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
import FillArrays: Fill
import UnPack: @unpack
import Gridap.ReferenceFEs: get_order
import GridapROMs.ParamDataStructures: GenericTransientRealisation, TransientRealisationAt
import GridapROMs.RBSteady: num_centroids,get_lhs,get_rhs,_get_label

export HighDimReduction
export SteadyReduction
export TransientReduction
export KroneckerReduction
export SequentialReduction
export HighDimHyperReduction
export SteadyHyperReduction
export TransientHyperReduction
export HighDimMDEIMHyperReduction
export HighDimSOPTHyperReduction
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

export TransientGreedyInterpolation
export TransientRBFInterpolation
export TransientBlockInterpolation
include("Interpolations.jl")

include("HyperReductions.jl")

include("HRAssemblers.jl")

include("LocalProjections.jl")

export TransientRBOperator
include("ReducedOperators.jl")

export SpaceTimeParamOperator
export SpaceTimeSolver
include("SpaceTime.jl")

include("ParamTimeMarching.jl")

include("PostProcess.jl")

include("Extensions.jl")

end # module
