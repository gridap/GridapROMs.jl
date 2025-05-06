struct UnCommonParamOperator{O,T,S} <: ParamOperator{O,T}
  operators::Vector{S}
end

ParamDataStructures.param_length(op::UnCommonParamOperator) = length(op.operators)

@inline function param_operator(f,args...)
  operators = map(f,args...)
  UnCommonParamOperator(operators)
end

function Algebra.solve(solver::ExtensionSolver,op::UnCommonParamOperator)
  op_batch = batchseries(op)
  x = pmap(op -> batchsolve(solver,op),op_batch)
  _to_consecutive(x)
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

function batchsolve(solver,op::UnCommonParamOperator)
  x = allocate_batchvector(op)
  for i in 1:length(op.operators)
    xi = solve(solver,op.operators[i])
    setindex!(x,xi,i)
  end
  return x
end

for (f,g,h) in zip(
  (:batchresidual,:batchjacobian),
  (:residual,:jacobian),
  (:allocate_batchvector,:allocate_batchmatrix)
  )
  @eval begin
    function $f(op::UnCommonParamOperator,x::AbstractParamVector)
      y = $h(op)
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
  UnCommonParamOperator(a.operators[pini:pend])
end

function _get_batch(a::ConsecutiveParamArray{T,N},pini,pend) where {T,N}
  data = view(a.data,_ncolons(Val{N}())...,pini:pend)
  ConsecutiveParamArray(data)
end

function _get_batch(a::Realization,pini,pend)
  Realization(a.params[pini:pend])
end

# function _get_batch(a::GenericTransientRealization,pini,pend)
#   GenericTransientRealization(_get_batch(a.params),a.times,a.t0)
# end
