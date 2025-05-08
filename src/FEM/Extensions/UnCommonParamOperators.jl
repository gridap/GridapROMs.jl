struct UnCommonParamOperator{O<:UnEvalOperatorType,T<:NonlinearDomainOperator} <: ParamOperator{O,JointDomains}
  operators::Vector{T}
  μ::AbstractRealization

  function UnCommonParamOperator(
    operators::Vector{T},
    μ::Realization
    ) where {T<:NonlinearDomainOperator{LinearEq}}

    @check param_length(μ) == length(operators)
    new{LinearParamEq,T}(operators,μ)
  end

  function UnCommonParamOperator(
    operators::Vector{T},
    μ::Realization
    ) where {T<:NonlinearDomainOperator{NonlinearEq}}

    @check param_length(μ) == length(operators)
    new{NonlinearParamEq,T}(operators,μ)
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
  op_batch = batchseries(op)
  t = @timed x = pmap(op -> batchsolve(solver,op),op_batch)
  stats = CostTracker(t,name="Solver";nruns=param_length(op))
  _to_consecutive(x),stats
end

function Algebra.residual(op::UnCommonParamOperator,x::AbstractParamVector)
  op_batch = batchseries(op)
  x_batch = batchseries(x)
  res = pmap((op,x) -> batchresidual(op,x),op_batch,x_batch)
  _to_consecutive(res)
end

function Algebra.jacobian(op::UnCommonParamOperator,x::AbstractParamVector)
  op_batch = batchseries(op)
  x_batch = batchseries(x)
  jac = pmap((op,x) -> batchjacobian(op,x),op_batch,x_batch)
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

for (f,g,h,k) in zip(
  (:batchresidual,:batchjacobian),
  (:residual,:jacobian),
  (:allocate_batchvector,:allocate_batchmatrix),
  (:allocate_firstresidual,:allocate_firstjacobian),
  )
  @eval begin
    function $f(op::UnCommonParamOperator,x::AbstractParamVector)
      y = $h(op)
      cache = $k(op)
      for i in 1:length(op.operators)
        xi = param_getindex(x,i)
        yi = $g(op.operators[i],xi)
        setindex!(y,yi,i)
      end
      return y
    end
  end
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

function batchseries(a;nbatches=nworkers())
  batchsize = floor(Int,param_length(a) / nbatches)
  ptrs = fill(Int32(batchsize),nbatches+1)
  for i in 1:mod(param_length(a),nbatches)
    ptrs[i] += 1
  end
  length_to_ptrs!(ptrs)
  batchseries(a,ptrs)
end

function batchseries(a,ptrs::Vector{<:Integer})
  T = _batchtype(a,ptrs[2],ptrs[1])
  batches = Vector{T}(undef,length(ptrs)-1)
  o = one(eltype(ptrs))
  for i in 1:length(ptrs)-1
    pini = ptrs[i]
    pend = ptrs[i+1]-o
    batches[i] = _get_batch(a,pini,pend)
  end
  return batches
end

_to_consecutive(x::GenericParamVector) = ConsecutiveParamArray(x)
_to_consecutive(x::BlockParamVector) = mortar(map(_to_consecutive,x.data))
_to_consecutive(x::Vector{<:AbstractParamArray}) = _to_consecutive(param_cat(x))

_vector_type(op::NonlinearOperator) = get_vector_type(get_test(op))
_vector_type(op::UnCommonParamOperator) = _vector_type(first(op.operators))

_num_dofs(f::SingleFieldFESpace) = num_free_dofs(f)
_num_dofs(f::DirectSumFESpace) = num_free_dofs(Extensions.get_bg_space(f))
_num_dofs(f::MultiFieldFESpace) = sum(num_free_dofs(fs) for fs in f)
_num_dofs(op::NonlinearOperator) = _num_dofs(get_test(op))

_num_dofs_at_field(op::NonlinearOperator,n::Int) = _num_dofs(get_test(op)[n])

_batchtype(a,pini,pend) = typeof(_get_batch(a,pini,pend))

function _get_batch(a::UnCommonParamOperator,pini,pend)
  UnCommonParamOperator(a.operators[pini:pend],a.μ[pini:pend])
end

function _get_batch(a::ConsecutiveParamArray{T,N},pini,pend) where {T,N}
  data = view(a.data,_ncolons(Val{N}())...,pini:pend)
  ConsecutiveParamArray(data)
end

# TODO write this properly
function _get_at_param(op::UnCommonParamOperator,μ::AbstractRealization)
  l = param_length(get_params(μ))
  UnCommonParamOperator(op.operators[1:l],μ)
end
