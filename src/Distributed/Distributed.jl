module Distributed

using DrWatson
using LinearAlgebra
using SparseArrays
using Serialization
using PartitionedArrays

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
using GridapDistributed

using GridapROMs.Utils
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.RBSteady

import Gridap.Helpers: @check

import GridapDistributed: DistributedSingleFieldFESpace
import PartitionedArrays: VectorAssemblyCache

const OPTIONS_CG_JACOBI = "-pc_type jacobi -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_CG_AMG = "-pc_type gamg -ksp_type cg -ksp_converged_reason -ksp_rtol 1.0e-10"
const OPTIONS_MUMPS = "-pc_type lu -ksp_type preonly -ksp_converged_reason -pc_factor_mat_solver_type mumps"
const OPTIONS_MINRES = "-ksp_type minres -ksp_converged_reason -ksp_rtol 1.0e-10"

export OPTIONS_CG_JACOBI,OPTIONS_CG_AMG,OPTIONS_MUMPS,OPTIONS_NEUTON_MUMPS,OPTIONS_MINRES

include("ParamAlgebra.jl")

include("ParamArrays.jl")

include("ParamSparseUtils.jl")

include("Primitives.jl")

include("ParamFESpaces.jl")
end
