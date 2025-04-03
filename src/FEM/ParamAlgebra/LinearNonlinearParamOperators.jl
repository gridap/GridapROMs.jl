"""
    get_linear_operator(op::NonlinearParamOperator) -> NonlinearParamOperator

Returns the linear part of the operator `op`
"""
get_linear_operator(op::NonlinearParamOperator) = @abstractmethod

"""
    get_nonlinear_operator(op::NonlinearParamOperator) -> NonlinearParamOperator

Returns the nonlinear part of the operator `op`
"""
get_nonlinear_operator(op::NonlinearParamOperator) = @abstractmethod

"""
    get_linear_systemcache(op::NonlinearParamOperator) -> AbstractParamCache

Returns the cache associated to the linear part of the operator `op`
"""
get_linear_systemcache(op::NonlinearParamOperator) = @abstractmethod

"""
    struct LinNonlinParamOperator{A,B} <: NonlinearParamOperator
      op_linear::A
      op_nonlinear::B
      cache_linear::AbstractParamCache
    end

Interface that allows to split the linear part of a parametric, nonlinear, differential
operator from the nonlinear part.
Fields:
- `op_linear`: linear part of the operator
- `op_nonlinear`: nonlinear part of the operator
- `cache_linear`: cache related to the linear part of the operator
"""
struct LinNonlinParamOperator{A,B} <: NonlinearParamOperator
  op_linear::A
  op_nonlinear::B
  cache_linear::AbstractParamCache
end

get_linear_operator(op::LinNonlinParamOperator) = op.op_linear
get_nonlinear_operator(op::LinNonlinParamOperator) = op.op_nonlinear
get_linear_systemcache(op::LinNonlinParamOperator) = op.cache_linear

function allocate_paramcache(op::LinNonlinParamOperator,μ::AbstractRealization)
  op_nlin = get_nonlinear_operator(op)
  allocate_paramcache(op_nlin,μ)
end

function allocate_systemcache(op::LinNonlinParamOperator,x::AbstractVector)
  cache_linear = get_linear_systemcache(op)
  op_nlin = get_nonlinear_operator(op)
  cache_nonlinear = allocate_systemcache(op_nlin,x)
  compatible_cache(cache_nonlinear,cache_linear)
end

function update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinNonlinParamOperator,
  μ::AbstractRealization)

  op_nlin = get_nonlinear_operator(op)
  update_paramcache!(paramcache,op_nlin,μ)
end

function update_systemcache!(op::LinNonlinParamOperator,x::AbstractVector)
  op_lin = get_linear_operator(op)
  syscache_lin = get_linear_systemcache(op)
  update_systemcache!(syscache_lin,op_lin,x)
end

function Algebra.allocate_residual(
  op::LinNonlinParamOperator,
  x::AbstractVector)

  syscache_lin = get_linear_systemcache(op)
  b_lin = get_vector(syscache_lin)
  similar(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinNonlinParamOperator,
  x::AbstractVector)

  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)
  similar(A_lin)
end

function Algebra.residual!(
  b::AbstractVector,
  op::LinNonlinParamOperator,
  x::AbstractVector)

  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)
  b_lin = get_vector(syscache_lin)

  op_nlin = get_nonlinear_operator(op)
  residual!(b,op_nlin,x)
  mul!(b,A_lin,x,1,1)
  axpy!(1,b_lin,b)

  b
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::LinNonlinParamOperator,
  x::AbstractVector)

  syscache_lin = get_linear_systemcache(op)
  A_lin = get_matrix(syscache_lin)

  op_nlin = get_nonlinear_operator(op)
  jacobian!(A,op_nlin,x)
  axpy!(1,A_lin,A)

  A
end

function Algebra.zero_initial_guess(op::LinNonlinParamOperator)
  v = get_vector(op.cache_linear)
  copy(v)
end

# utils

function compatible_cache(a::SystemCache,b::SystemCache)
  A′ = compatible_cache(a.A,b.A)
  b′ = compatible_cache(a.b,b.b)
  SystemCache(A′,b′)
end

function compatible_cache(a::AbstractParamArray,b::AbstractParamArray)
  a
end

function compatible_cache(a::ParamSparseMatrix,b::ParamSparseMatrix)
  ia,ja,_ = findnz(a)
  ib,jb,_ = findnz(b)
  if ia == ib && ja == jb
    a
  else
    copy(b)
  end
end

function compatible_cache(ba::BlockParamArray,bb::BlockParamArray)
  @check size(ba) == size(bb)
  ba′ = map(blocks(ba),blocks(bb)) do a,b
    compatible_cache(a,b)
  end
  mortar(ba′)
end
