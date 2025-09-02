module ParamODEs

using LinearAlgebra
using ForwardDiff

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Helpers

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady

import BlockArrays: blocks,blocklength,mortar
import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ODEs: TransientCellField
import Gridap.ReferenceFEs: get_order
import GridapROMs.ParamSteady: get_domains_res,get_domains_jac
import GridapROMs.Utils: change_domains,set_domains

export ∂ₚt,∂ₚtt
include("TimeDerivatives.jl")

include("TransientParamCellFields.jl")

export TransientTrialParamFESpace
export TransientMultiFieldParamFESpace
include("TransientTrialParamFESpaces.jl")

include("TransientNonlinearParamOperators.jl")

export ODEParamOperatorType
export NonlinearParamODE
export LinearParamODE
export LinearNonlinearParamODE
export ODEParamOperator
export JointODEParamOperator
export SplitODEParamOperator
export LinearNonlinearODEParamOperator
export TransientLinearParamOperator
export TransientParamOperator
export LinearNonlinearTransientParamOperator
include("ODEParamOperators.jl")

export ParamStageOperator
include("ParamStageOperators.jl")

export TransientParamFEOperator
export SplitTransientParamFEOperator
export JointTransientParamFEOperator
export TransientParamFEOpFromWeakForm
export TransientLinearParamFEOperator
export TransientLinearParamFEOpFromWeakForm
include("TransientParamFEOperators.jl")

include("ParamTimeMarching.jl")

export ShiftedSolver
include("ShiftedSolvers.jl")

export ODEParamSolution
export initial_condition
include("ODEParamSolutions.jl")

include("TransientParamFESolutions.jl")

end # module
