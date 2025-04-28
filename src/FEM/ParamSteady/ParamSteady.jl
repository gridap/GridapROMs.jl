module ParamSteady

using LinearAlgebra
using BlockArrays
using SparseArrays

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.Fields
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.MultiField
using Gridap.ODEs
using Gridap.Helpers

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers
using GridapSolvers.MultilevelTools

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces

import Test: @test
import Gridap.Algebra: residual!,jacobian!
import Gridap.FESpaces: FEFunction,SparseMatrixAssembler,EvaluationFunction
import Gridap.ReferenceFEs: get_order
import GridapROMs.Utils: CostTracker,change_domains,set_domains

export UnEvalTrialFESpace
export ParamTrialFESpace
export AbstractTrialFESpace
include("ParamTrialFESpaces.jl")

export UnEvalOperatorType
export NonlinearParamEq
export LinearParamEq
export LinearNonlinearParamEq
export TriangulationStyle
export JointDomains
export SplitDomains
export ParamOperator
export JointParamOperator
export SplitParamOperator
export LinearParamOperator
export LinearNonlinearParamOperator
export get_fe_operator
include("ParamOperators.jl")

export ParamFEOperator
export SplitParamFEOperator
export JointParamFEOperator
export LinearParamFEOperator
export FEDomains
export get_jac
export get_param_space
export get_param_assembler
export get_domains
export get_domains_res
export get_domains_jac
export get_dof_map_at_domains
export get_sparse_dof_map_at_domains
include("ParamFEOperators.jl")

export LinearNonlinearParamFEOperator
export join_operators
include("LinearNonlinearParamFEOperators.jl")

export GenericParamOperator
export GenericLinearNonlinearParamOperator
export collect_cell_matrix_for_trian
export collect_cell_vector_for_trian
include("ParamOpFromFEOps.jl")

include("ParamFESolvers.jl")

end # module
