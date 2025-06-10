module Uncommon

export UncommonParamOperator
export param_operator
export allocate_batchvector
export allocate_batchmatrix

using BlockArrays
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

using GridapROMs.Utils
using GridapROMs.DofMaps
using GridapROMs.TProduct
using GridapROMs.ParamDataStructures
using GridapROMs.ParamAlgebra
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady
using GridapROMs.ParamODEs
using GridapROMs.Extensions

struct UncommonParamOperator{O<:UnEvalOperatorType,T<:TriangulationStyle} <: ParamOperator{O,T}
  operators::Vector{<:DomainOperator}
  domains::FEDomains
  μ::AbstractRealization

  function UncommonParamOperator{O,T}(
    operators::Vector{<:DomainOperator},
    domains::FEDomains,
    μ::AbstractRealization
    ) where {O,T}

    @check param_length(μ) == length(operators)
    new{O,T}(operators,domains,μ)
  end
end

const JointUncommonParamOperator{O<:UnEvalOperatorType} = UncommonParamOperator{O,JointDomains}

const SplitUncommonParamOperator{O<:UnEvalOperatorType} = UncommonParamOperator{O,SplitDomains}

function UncommonParamOperator(
  operators::Vector{<:DomainOperator{LinearEq}},
  μ::AbstractRealization
  )

  domains = FEDomains()
  UncommonParamOperator{NonlinearParamEq,JointDomains}(operators,domains,μ)
end

function UncommonParamOperator(
  operators::Vector{<:DomainOperator{LinearEq}},
  μ::AbstractRealization,
  domains::FEDomains
  )

  UncommonParamOperator{LinearParamEq,SplitDomains}(operators,domains,μ)
end

function UncommonParamOperator(
  operators::Vector{<:DomainOperator{NonlinearEq}},
  μ::AbstractRealization)

  domains = FEDomains()
  UncommonParamOperator{NonlinearParamEq,JointDomains}(operators,domains,μ)
end

function UncommonParamOperator(
  operators::Vector{<:DomainOperator{NonlinearEq}},
  μ::AbstractRealization,
  domains::FEDomains)

  UncommonParamOperator{NonlinearParamEq,SplitDomains}(operators,domains,μ)
end

function UncommonParamOperator(
  operators::Vector{<:DomainOperator{O,SplitDomains}},
  args...) where O<:OperatorType

  @notimplemented "The input FE operators must not be split on multiple domains"
end

@inline function param_operator(f,μ::AbstractRealization,args...)
  operators = map(f,μ)
  UncommonParamOperator(operators,μ,args...)
end

ParamDataStructures.param_length(op::UncommonParamOperator) = param_length(op.μ)

function ParamDataStructures.realization(op::UncommonParamOperator;nparams=1)
  @assert nparams ≤ param_length(op)
  op.μ[1:nparams]
end

FESpaces.get_test(op::UncommonParamOperator) = get_bg_space(get_test(first(op.operators)))
FESpaces.get_trial(op::UncommonParamOperator) = get_bg_space(get_trial(first(op.operators)))

Utils.get_domains(op::UncommonParamOperator) = op.domains
Utils.get_domains_res(op::UncommonParamOperator) = get_domains_res(get_domains(op))
Utils.get_domains_jac(op::UncommonParamOperator) = get_domains_jac(get_domains(op))

function Utils.set_domains(op::UncommonParamOperator{O}) where O
  UncommonParamOperator{O,JointDomains}(op.operators,op.domains,op.μ)
end

function Utils.change_domains(op::UncommonParamOperator{O,T},args...) where {O,T}
  domains = FEDomains(args...)
  UncommonParamOperator{O,T}(op.operators,domains,op.μ)
end

DofMaps.get_dof_map(op::UncommonParamOperator) = get_dof_map(get_test(op))
DofMaps.get_sparse_dof_map(op::UncommonParamOperator) = get_sparse_dof_map(get_trial(op),get_test(op))

FESpaces.assemble_matrix(op::UncommonParamOperator,form::Function) = ParamSteady._assemble_matrix(form,get_test(op))

function Algebra.solve(solver::ExtensionSolver,op::UncommonParamOperator)
  t = @timed x = batchsolve(solver,op)
  stats = CostTracker(t,name="Solver";nruns=param_length(op))
  _to_consecutive(x),stats
end

function Algebra.residual(op::UncommonParamOperator,x::AbstractParamVector)
  res = batchresidual(op,x)
  _to_consecutive(res)
end

function Algebra.jacobian(op::UncommonParamOperator,x::AbstractParamVector)
  jac = batchjacobian(op,x)
  _to_consecutive(jac)
end

function Algebra.solve(solver::NonlinearSolver,op::UncommonParamOperator,μ::Realization)
  solve(solver,_get_at_param(op,μ))
end

function Algebra.residual(op::UncommonParamOperator,μ::Realization,x::AbstractParamVector)
  residual(_get_at_param(op,μ),x)
end

function Algebra.jacobian(op::UncommonParamOperator,μ::Realization,x::AbstractParamVector)
  jacobian(_get_at_param(op,μ),x)
end

function batchsolve(solver,op::UncommonParamOperator)
  x = allocate_batchvector(op)
  for i in 1:length(op.operators)
    xi = solve(solver,op.operators[i])
    setindex!(x,xi,i)
  end
  return x
end

function batchresidual(op::UncommonParamOperator,x::AbstractParamVector)
  b = allocate_batchvector(op)
  for i in 1:length(op.operators)
    xi = param_getindex(x,i)
    bi = residual(op.operators[i],xi)
    setindex!(b,bi,i)
  end
  return b
end

function batchjacobian(op::UncommonParamOperator,x::AbstractParamVector)
  cache = allocate_batchmatrix(op)
  for i in 1:length(op.operators)
    xi = param_getindex(x,i)
    Ai = jacobian(op.operators[i],xi)
    update_batchmatrix!(cache,Ai)
  end
  return to_param_sparse_matrix(cache)
end

function batchresidual(op::SplitUncommonParamOperator,x::AbstractParamVector)
  b = batchresidual(set_domains(op),x)
  contribution(get_domains_res(op)) do trian
    b
  end
end

function batchjacobian(op::SplitUncommonParamOperator,x::AbstractParamVector)
  A = batchjacobian(set_domains(op),x)
  contribution(get_domains_jac(op)) do trian
    A
  end
end

# utils

function allocate_batchvector(op::UncommonParamOperator)
  allocate_batchvector(_vector_type(op),op)
end

function allocate_batchvector(::Type{V},op::UncommonParamOperator) where V
  plength = length(op.operators)
  ptrs = Vector{Int}(undef,plength+1)
  for i in 1:plength
    ptrs[i+1] = _num_dofs(op.operators[i])
  end
  length_to_ptrs!(ptrs)
  data = allocate_vector(V,ptrs[end]-1)
  GenericParamVector(data,ptrs)
end

function allocate_batchvector(::Type{V},op::UncommonParamOperator) where V<:BlockVector
  plength = length(op.operators)
  nfields = num_fields(get_test(first(op.operators)))
  data = map(1:nfields) do n
    ptrs = Vector{Int}(undef,plength+1)
    for i in 1:plength
      ptrs[i+1] = _num_dofs_at_field(op.operators[i],n)
    end
    length_to_ptrs!(ptrs)
    data = allocate_vector(V,ptrs[end]-1)
    GenericParamVector(data,ptrs)
  end
  mortar(data)
end

function allocate_batchmatrix(op::UncommonParamOperator)
  allocate_batchmatrix(_vector_type(op),op)
end

function allocate_batchmatrix(::Type{V},op::UncommonParamOperator) where V
  plength = length(op.operators)
  n = 0
  for i in 1:plength
    n += _num_dofs_test_trial(op.operators[i])
  end
  colptr = Vector{Int}(undef,n)
  rowval = Vector{Int}(undef,n)
  data = Vector{eltype(V)}(undef,n)
  ptrs = Vector{Int}(undef,plength+1)
  s = [0,0]
  counts = [0,0,1]
  return (s,colptr,rowval,data,ptrs,counts)
end

function allocate_batchmatrix(::Type{V},op::UncommonParamOperator) where V<:BlockVector
  plength = length(op.operators)
  nfields_test = num_fields(get_test(first(op.operators)))
  nfields_trial = num_fields(get_trial(first(op.operators)))
  array = map(CartesianIndices((nfields_test,nfields_trial))) do i,j
    n = 0
    for k in 1:plength
      n += _num_dofs_test_trial_at_field(op.operators[k],i,j)
    end
    colptr = Vector{Int}(undef,n)
    rowval = Vector{Int}(undef,n)
    data = Vector{eltype(V)}(undef,n)
    ptrs = Vector{Int}(undef,plength+1)
    s = zeros(Int,2)
    counts = zeros(Int,3)
    (s,colptr,rowval,data,ptrs,counts)
  end
  touched = fill(true,size(array))
  ArrayBlock(array,touched)
end

function update_batchmatrix!(cache,mat::SparseMatrixCSC)
  s,colptr,rowval,data,ptrs,counts = cache
  s .= size(mat)
  i,j,k = counts
  ptrs[k+1] = nnz(mat)
  for l in 1:nnz(mat)
    rowval[i+l] = mat.rowval[l]
    data[i+l] = mat.nzval[l]
    if l ≤ length(mat.colptr)
      colptr[j+l] = mat.colptr[l]
    end
  end
  counts .+= (nnz(mat),length(mat.colptr),1)
  return
end

function update_batchmatrix!(cache::ArrayBlock,mat::BlockMatrix)
  for i in eachindex(cache)
    if cache.touched[i]
      update_batchmatrix!(cache.array[i],blocks(mat)[i])
    end
  end
end

function to_param_sparse_matrix(cache)
  s,colptr,rowval,data,ptrs,counts = cache
  nr,nc = s
  i,j,k = counts
  resize!(colptr,j)
  resize!(rowval,i)
  resize!(data,i)
  resize!(ptrs,k)
  length_to_ptrs!(ptrs)
  GenericParamSparseMatrixCSC(nr,nc,colptr,rowval,data,ptrs)
end

function to_param_sparse_matrix(cache::ArrayBlock)
  mortar(map(to_param_sparse_matrix,cache.array))
end

_to_consecutive(x::GenericParamVector) = ConsecutiveParamArray(x)
_to_consecutive(x::BlockParamVector) = mortar(map(_to_consecutive,x.data))
_to_consecutive(x::GenericParamSparseMatrixCSC) = ConsecutiveParamSparseMatrixCSC(x)

function _to_consecutive(x::Contribution)
  contribution(get_domains(x)) do trian
    _to_consecutive(x[trian])
  end
end

_vector_type(op::NonlinearOperator) = get_vector_type(get_test(op))
_vector_type(op::UncommonParamOperator) = _vector_type(first(op.operators))

_num_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
_num_dofs(f::DirectSumFESpace) = _num_dofs(get_bg_space(f))
_num_dofs(op::NonlinearOperator) = _num_dofs(get_test(op))

_num_dofs_test_trial(f::SingleFieldFESpace,g::SingleFieldFESpace) = num_free_dofs(f)*num_free_dofs(g)
function _num_dofs_test_trial(f::DirectSumFESpace,g::SingleFieldFESpace)
  _num_dofs_test_trial(get_bg_space(f),get_bg_space(g))
end
_num_dofs_test_trial(op::NonlinearOperator) = _num_dofs_test_trial(get_test(op),get_trial(op))

_num_dofs_at_field(op::NonlinearOperator,n::Int) = _num_dofs(get_test(op)[n])

function _num_dofs_test_trial_at_field(op::NonlinearOperator,m::Int,n::Int)
  _num_dofs_test_trial(get_test(op)[m],get_trial(op)[n])
end

# TODO write this properly
function _get_at_param(op::UncommonParamOperator,μ::AbstractRealization)
  l = param_length(get_params(μ))
  UncommonParamOperator(op.operators[1:l],μ,op.domains)
end

end # end module
