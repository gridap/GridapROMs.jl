struct UnCommonParamOperator{O<:UnEvalOperatorType} <: ParamOperator{O,JointDomains}
  operators::Vector{<:DomainOperator}
  μ::AbstractRealization

  function UnCommonParamOperator(
    operators::Vector{<:DomainOperator{LinearEq}},
    μ::Realization
    )

    @check param_length(μ) == length(operators)
    new{LinearParamEq}(operators,μ)
  end

  function UnCommonParamOperator(
    operators::Vector{<:DomainOperator{NonlinearEq}},
    μ::Realization
    )

    @check param_length(μ) == length(operators)
    new{NonlinearParamEq}(operators,μ)
  end
end

ParamDataStructures.param_length(op::UnCommonParamOperator) = param_length(op.μ)

function ParamDataStructures.realization(op::UnCommonParamOperator;nparams=1)
  @assert nparams ≤ param_length(op)
  op.μ[1:nparams]
end

FESpaces.get_test(op::UnCommonParamOperator) = get_bg_space(get_test(first(op.operators)))
FESpaces.get_trial(op::UnCommonParamOperator) = get_bg_space(get_trial(first(op.operators)))

DofMaps.get_dof_map(op::UnCommonParamOperator) = get_dof_map(get_test(op))
DofMaps.get_sparse_dof_map(op::UnCommonParamOperator) = get_sparse_dof_map(get_trial(op),get_test(op))

FESpaces.assemble_matrix(op::UnCommonParamOperator,form::Function) = ParamSteady._assemble_matrix(form,get_test(op))

@inline function param_operator(f,μ::AbstractRealization)
  operators = map(f,μ)
  UnCommonParamOperator(operators,μ)
end

function Algebra.solve(solver::ExtensionSolver,op::UnCommonParamOperator)
  t = @timed x = batchsolve(solver,op)
  stats = CostTracker(t,name="Solver";nruns=param_length(op))
  _to_consecutive(x),stats
end

function Algebra.residual(op::UnCommonParamOperator,x::AbstractParamVector)
  res = batchresidual(op,x)
  _to_consecutive(res)
end

function Algebra.jacobian(op::UnCommonParamOperator,x::AbstractParamVector)
  jac = batchjacobian(op,x)
  _to_consecutive(jac)
end

function Algebra.solve(solver::NonlinearSolver,op::UnCommonParamOperator,μ::Realization)
  solve(solver,_get_at_param(op,μ))
end

function Algebra.residual(op::UnCommonParamOperator,μ::Realization,x::AbstractParamVector)
  residual(_get_at_param(op,μ),x)
end

function Algebra.jacobian(op::UnCommonParamOperator,μ::Realization,x::AbstractParamVector)
  jacobian(_get_at_param(op,μ),x)
end

function batchsolve(solver,op::UnCommonParamOperator)
  x = allocate_batchvector(op)
  for i in 1:length(op.operators)
    xi = solve(solver,op.operators[i])
    setindex!(x,xi,i)
  end
  return x
end

function batchresidual(op::UnCommonParamOperator,x::AbstractParamVector)
  b = allocate_batchvector(op)
  for i in 1:length(op.operators)
    xi = param_getindex(x,i)
    bi = residual(op.operators[i],xi)
    setindex!(b,bi,i)
  end
  return b
end

function batchjacobian(op::UnCommonParamOperator,x::AbstractParamVector)
  cache = allocate_batchmatrix(op)
  for i in 1:length(op.operators)
    xi = param_getindex(x,i)
    Ai = jacobian(op.operators[i],xi)
    update_batchmatrix!(cache,Ai)
  end
  return to_param_sparse_matrix(cache)
end

# utils

function allocate_batchvector(op::UnCommonParamOperator)
  allocate_batchvector(_vector_type(op),op)
end

function allocate_batchvector(::Type{V},op::UnCommonParamOperator) where V
  plength = length(op.operators)
  ptrs = Vector{Int}(undef,plength+1)
  for i in 1:plength
    ptrs[i+1] = _num_dofs(op.operators[i])
  end
  length_to_ptrs!(ptrs)
  data = allocate_vector(V,ptrs[end]-1)
  GenericParamVector(data,ptrs)
end

function allocate_batchvector(::Type{V},op::UnCommonParamOperator) where V<:BlockVector
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

function allocate_batchmatrix(op::UnCommonParamOperator)
  allocate_batchmatrix(_vector_type(op),op)
end

function allocate_batchmatrix(::Type{V},op::UnCommonParamOperator) where V
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

function allocate_batchmatrix(::Type{V},op::UnCommonParamOperator) where V<:BlockVector
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

_vector_type(op::NonlinearOperator) = get_vector_type(get_test(op))
_vector_type(op::UnCommonParamOperator) = _vector_type(first(op.operators))

_num_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
_num_dofs(f::DirectSumFESpace) = _num_dofs(Extensions.get_bg_space(f))
_num_dofs(op::NonlinearOperator) = _num_dofs(get_test(op))

_num_dofs_test_trial(f::SingleFieldFESpace,g::SingleFieldFESpace) = num_free_dofs(f)*num_free_dofs(g)
function _num_dofs_test_trial(f::DirectSumFESpace,g::SingleFieldFESpace)
  _num_dofs_test_trial(Extensions.get_bg_space(f),Extensions.get_bg_space(g))
end
_num_dofs_test_trial(op::NonlinearOperator) = _num_dofs_test_trial(get_test(op),get_trial(op))

_num_dofs_at_field(op::NonlinearOperator,n::Int) = _num_dofs(get_test(op)[n])

function _num_dofs_test_trial_at_field(op::NonlinearOperator,m::Int,n::Int)
  _num_dofs_test_trial(get_test(op)[m],get_trial(op)[n])
end

# TODO write this properly
function _get_at_param(op::UnCommonParamOperator,μ::AbstractRealization)
  l = param_length(get_params(μ))
  UnCommonParamOperator(op.operators[1:l],μ)
end
