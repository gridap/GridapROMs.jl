module Uncommon

export UncommonParamOperator
export UncommonContribution
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
  μ::AbstractRealization

  function UncommonParamOperator(
    operators::Vector{<:DomainOperator{LinearEq,T}},
    μ::AbstractRealization
    ) where T

    @check param_length(μ) == length(operators)
    new{LinearParamEq,T}(operators,μ)
  end

  function UncommonParamOperator(
    operators::Vector{<:DomainOperator{NonlinearEq,T}},
    μ::AbstractRealization
    ) where T

    @check param_length(μ) == length(operators)
    new{NonlinearParamEq,T}(operators,μ)
  end
end

const JointUncommonParamOperator{O<:UnEvalOperatorType} = UncommonParamOperator{O,JointDomains}

const SplitUncommonParamOperator{O<:UnEvalOperatorType} = UncommonParamOperator{O,SplitDomains}

@inline function param_operator(f,μ::AbstractRealization)
  operators = map(f,μ)
  UncommonParamOperator(operators,μ)
end

ParamDataStructures.param_length(op::UncommonParamOperator) = param_length(op.μ)

function ParamDataStructures.realization(op::UncommonParamOperator;nparams=1)
  @assert nparams ≤ param_length(op)
  op.μ[1:nparams]
end

FESpaces.get_test(op::UncommonParamOperator) = get_bg_space(get_test(first(op.operators)))
FESpaces.get_trial(op::UncommonParamOperator) = parameterize(get_bg_space(get_trial(first(op.operators))),1)

Utils.change_domains(op::UncommonParamOperator,args...) = op
Utils.set_domains(op::SplitUncommonParamOperator) = UncommonParamOperator(set_domains.(op.operators),op.μ)

DofMaps.get_dof_map(op::UncommonParamOperator) = get_dof_map(get_test(op))
DofMaps.get_sparse_dof_map(op::UncommonParamOperator) = get_sparse_dof_map(get_trial(op),get_test(op))

FESpaces.assemble_matrix(op::UncommonParamOperator,form::Function) = ParamSteady._assemble_matrix(form,get_test(op))

function Algebra.solve(solver::ExtensionSolver,op::JointUncommonParamOperator)
  t = @timed x = batchsolve(solver,op)
  stats = CostTracker(t,name="Solver";nruns=param_length(op))
  _to_consecutive(x),stats
end

function Algebra.solve(solver::ExtensionSolver,op::SplitUncommonParamOperator)
  solve(solver,set_domains(op))
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
    update_batchvector!(b,bi,i)
  end
  return b
end

function batchjacobian(op::UncommonParamOperator,x::AbstractParamVector)
  A = allocate_batchmatrix(op)
  for i in 1:length(op.operators)
    xi = param_getindex(x,i)
    Ai = jacobian(op.operators[i],xi)
    update_batchmatrix!(A,Ai,i)
  end
  return to_param_sparse_matrix(A)
end

# utils

for f in (:allocate_batchvector,:allocate_batchmatrix)
  g = f==:allocate_batchvector ? :get_domains_res : :get_domains_jac
  @eval begin
    function $f(op::JointUncommonParamOperator)
      $f(_vector_type(op),op)
    end

    function $f(op::SplitUncommonParamOperator)
      trians = $g(first(op.operators))
      contribution(trians) do trian
        $f(_vector_type(op),op)
      end
    end
  end
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

function allocate_batchvector(::Type{V},op::UncommonParamOperator) where {T,V<:BlockVector{T}}
  plength = length(op.operators)
  nfields = num_fields(get_test(first(op.operators)))
  data = map(1:nfields) do n
    ptrs = Vector{Int}(undef,plength+1)
    for i in 1:plength
      ptrs[i+1] = _num_dofs_at_field(op.operators[i],n)
    end
    length_to_ptrs!(ptrs)
    data = allocate_vector(Vector{T},ptrs[end]-1)
    GenericParamVector(data,ptrs)
  end
  mortar(data)
end

function allocate_batchmatrix(::Type{V},op::UncommonParamOperator) where V
  plength = length(op.operators)
  n = 2*plength*_max_num_nonzeros(op)
  colptr = Vector{Int}(undef,n)
  rowval = Vector{Int}(undef,n)
  data = Vector{eltype(V)}(undef,n)
  ptrs = Vector{Int}(undef,plength+1)
  s = [0,0]
  counts = [0,0]
  return (s,colptr,rowval,data,ptrs,counts)
end

function allocate_batchmatrix(::Type{V},op::UncommonParamOperator) where {T,V<:BlockVector{T}}
  plength = length(op.operators)
  nfields_test = num_fields(get_test(first(op.operators)))
  nfields_trial = num_fields(get_trial(first(op.operators)))
  array = map(CartesianIndices((nfields_test,nfields_trial))) do i,j
    n = 2*plength*_max_num_nonzeros(op,i,j)
    colptr = Vector{Int}(undef,n)
    rowval = Vector{Int}(undef,n)
    data = Vector{T}(undef,n)
    ptrs = Vector{Int}(undef,plength+1)
    s = zeros(Int,2)
    counts = zeros(Int,2)
    (s,colptr,rowval,data,ptrs,counts)
  end
  touched = fill(true,size(array))
  ArrayBlock(array,touched)
end

function update_batchvector!(cache,vec::AbstractVector,k::Int)
  setindex!(cache,vec,k)
end

function update_batchmatrix!(cache,mat::SparseMatrixCSC,k::Int;α=1)
  s,colptr,rowval,data,ptrs,counts = cache
  s .= size(mat)
  i,j = counts
  ptrs[k+1] = nnz(mat)

  if length(rowval)-i < nnz(mat)
    @warn "Need to expand sparse structures size; consider allocating larger structures"
    resize!(rowval,length(rowval)+α*nnz(mat))
    resize!(data,length(rowval)+α*nnz(mat))
  end

  if length(colptr)-j < length(mat.colptr)
    @warn "Need to expand sparse structures size; consider allocating larger structures"
    resize!(colptr,length(colptr)+α*length(mat.colptr))
  end

  if nnz(mat) ≥ length(mat.colptr)
    for l in 1:nnz(mat)
      rowval[i+l] = mat.rowval[l]
      data[i+l] = mat.nzval[l]
      if l ≤ length(mat.colptr)
        colptr[j+l] = mat.colptr[l]
      end
    end
  else
    for l in 1:length(mat.colptr)
      colptr[j+l] = mat.colptr[l]
      if l ≤ nnz(mat)
        rowval[i+l] = mat.rowval[l]
        data[i+l] = mat.nzval[l]
      end
    end
  end

  counts .+= (nnz(mat),length(mat.colptr))
  return
end

for f in (:update_batchvector!,:update_batchmatrix!)
  @eval begin
    function $f(cache::ArrayBlock,matvec::BlockArray,k::Int)
      for i in eachindex(cache)
        if cache.touched[i]
          $f(cache.array[i],blocks(matvec)[i],k)
        end
      end
    end

    function $f(cache::Contribution,matvec::Contribution,k::Int)
      map(eachindex(matvec)) do i
        $f(cache[i],matvec[i],k)
      end
    end
  end
end

function to_param_sparse_matrix(cache)
  s,colptr,rowval,data,ptrs,counts = cache
  nr,nc = s
  i,j = counts
  resize!(colptr,j)
  resize!(rowval,i)
  resize!(data,i)
  length_to_ptrs!(ptrs)
  GenericParamSparseMatrixCSC(nr,nc,colptr,rowval,data,ptrs)
end

function to_param_sparse_matrix(cache::ArrayBlock)
  mortar(map(to_param_sparse_matrix,cache.array))
end

function to_param_sparse_matrix(cache::Contribution)
  contribution(get_domains(cache)) do trian
    to_param_sparse_matrix(cache[trian])
  end
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

function _max_num_nonzeros(f::SingleFieldFESpace,g::SingleFieldFESpace)
  sparsity = get_sparsity(f,g)
  nnz(sparsity)
end

function _max_num_nonzeros(f::DirectSumFESpace,g::SingleFieldFESpace)
  _max_num_nonzeros(get_bg_space(f),get_bg_space(g))
end

function _max_num_nonzeros(op::UncommonParamOperator)
  ndofs = map(opi -> num_free_dofs(get_test(opi)),op.operators)
  maxid = findfirst(ndofs .== maximum(ndofs))
  _max_num_nonzeros(get_trial(op.operators[maxid]),get_test(op.operators[maxid]))
end

_num_dofs_at_field(op::NonlinearOperator,n::Int) = _num_dofs(get_test(op)[n])

function _max_num_nonzeros_at_field(op::NonlinearOperator,m::Int,n::Int)
  _max_num_nonzeros(get_test(op)[m],get_trial(op)[n])
end

# TODO maybe use K means here as well?
function _get_at_param(op::UncommonParamOperator,μ::AbstractRealization)
  @assert all(μi == ξi for (μi,ξi) in zip(μ,op.μ))
  opμ = op.operators[1:num_params(μ)]
  UncommonParamOperator(opμ,μ)
end

end # end module
