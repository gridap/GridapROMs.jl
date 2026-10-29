"""
    abstract type NonlinearParamOperator <: NonlinearOperator end
"""
abstract type NonlinearParamOperator <: NonlinearOperator end

function Algebra.allocate_residual(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  @abstractmethod
end

function Algebra.residual!(
  b,
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  @abstractmethod
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector
  )

  paramcache = allocate_paramcache(nlop,μ)
  residual(nlop,μ,x,paramcache)
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  b = allocate_residual(nlop,μ,x,paramcache)
  residual!(b,nlop,μ,x,paramcache)
  b
end

function Algebra.allocate_jacobian(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  LinearAlgebra.fillstored!(A,false)
  jacobian_add!(A,nlop,μ,x,paramcache)
  A
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector
  )

  paramcache = allocate_paramcache(nlop,μ)
  jacobian(nlop,μ,x,paramcache)
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  μ::Realisation,
  x::AbstractVector,
  paramcache
  )

  A = allocate_jacobian(nlop,μ,x,paramcache)
  jacobian!(A,nlop,μ,x,paramcache)
  A
end

# caches

"""
    abstract type AbstractParamCache <: GridapType end
"""
abstract type AbstractParamCache <: GridapType end

"""
    allocate_paramcache(nlop::NonlinearParamOperator,μ::AbstractRealisation) -> AbstractParamCache

Similar to `allocate_odecache` in [`Gridap`](@ref), when dealing with parametric problems
"""
function allocate_paramcache(nlop::NonlinearParamOperator,μ::AbstractRealisation)
  @abstractmethod
end

"""
    update_paramcache!(paramcache::AbstractParamCache,nlop::NonlinearParamOperator,μ::AbstractRealisation) -> AbstractParamCache

Similar to `update_odecache!` in [`Gridap`](@ref), when dealing with parametric problems
"""
function update_paramcache!(
  paramcache::AbstractParamCache,
  nlop::NonlinearParamOperator,
  μ::AbstractRealisation
  )

  @abstractmethod
end

"""
    mutable struct ParamCache <: AbstractParamCache
      trial
      ptrial
    end
"""
mutable struct ParamCache <: AbstractParamCache
  trial
  ptrial
end

function update_paramcache!(
  c::ParamCache,
  nlop::NonlinearParamOperator,
  μ::AbstractRealisation
  )

  c.trial = evaluate!(c.trial,c.ptrial,μ)
  c
end

"""
    struct SystemCache <: AbstractParamCache
      A
      b
      x
    end
"""
struct SystemCache <: AbstractParamCache
  A
  b
  x
  function SystemCache(A,b,x)
    fill!(x,zero(eltype(x)))
    new(A,b,x)
  end
end

Algebra.get_matrix(c::SystemCache) = c.A
Algebra.get_vector(c::SystemCache) = c.b

"""
    allocate_systemcache(nlop::NonlinearParamOperator[,x::AbstractVector]) -> SystemCache

Allocates the Jacobian, residual and solution vector of `nlop`
"""
function allocate_systemcache(nlop::NonlinearParamOperator)
  @abstractmethod
end

function allocate_systemcache(nlop::NonlinearParamOperator,x::AbstractVector)
  A = allocate_jacobian(nlop,x)
  b = allocate_residual(nlop,x)
  y = similar(x)
  return SystemCache(A,b,y)
end

"""
    update_systemcache!(nlop::NonlinearParamOperator,x::AbstractVector)

Updates the system cache stored in `nlop`, if any. For a [`LinNonlinParamOperator`](@ref),
assembles its linear part
"""
function update_systemcache!(nlop::NonlinearParamOperator,x::AbstractVector)
  nothing 
end

Base.copy(c::SystemCache) = SystemCache(copy(c.A),copy(c.b),copy(c.x))
Base.similar(c::SystemCache) = SystemCache(similar(c.A),similar(c.b),similar(c.x))

"""
    struct GenericParamNonlinearOperator{A} <: NonlinearParamOperator
      op::A
      μ::AbstractRealisation
      paramcache::ParamCache
    end

Fields:
- `op`: `NonlinearParamOperator` representing a parametric differential problem
- `μ`: `AbstractRealisation` representing the parameters at which the problem is solved
- `paramcache`: cache of the problem
"""
struct GenericParamNonlinearOperator{A} <: NonlinearParamOperator
  op::A
  μ::AbstractRealisation
  paramcache::ParamCache
end

function ParamDataStructures.parameterise(op::NonlinearParamOperator,μ::AbstractRealisation)
  paramcache = allocate_paramcache(op,μ)
  GenericParamNonlinearOperator(op,μ,paramcache)
end

function Algebra.allocate_residual(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector
  )

  allocate_residual(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector
  )

  residual!(b,nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector
  )

  allocate_jacobian(nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::GenericParamNonlinearOperator,
  x::AbstractVector
  )

  jacobian!(A,nlop.op,nlop.μ,x,nlop.paramcache)
end

function Algebra.zero_initial_guess(nlop::GenericParamNonlinearOperator)
  zero_initial_guess(nlop.op,nlop.μ)
end

function allocate_systemcache(nlop::GenericParamNonlinearOperator)
  xh = zero(nlop.paramcache.trial)
  x = get_free_dof_values(xh)
  allocate_systemcache(nlop,x)
end

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

Utils.get_linear_part(op::LinNonlinParamOperator) = op.op_linear
Utils.get_nonlinear_part(op::LinNonlinParamOperator) = op.op_nonlinear

function allocate_paramcache(op::LinNonlinParamOperator,μ::AbstractRealisation)
  allocate_paramcache(op.op_nonlinear,μ)
end

function allocate_systemcache(op::LinNonlinParamOperator,x::AbstractVector)
  allocate_systemcache(op.op_nonlinear,x)
end

function update_paramcache!(
  paramcache::AbstractParamCache,
  op::LinNonlinParamOperator,
  μ::AbstractRealisation
  )

  update_paramcache!(paramcache,op.op_nonlinear,μ)
end

# x is not used for the res/jac updates; rather, the x (which is identically zero) stored 
# within the linear cache is used 
function update_systemcache!(op::LinNonlinParamOperator,x::AbstractVector)
  residual!(op.cache_linear.b,op.op_linear,op.cache_linear.x)
  jacobian!(op.cache_linear.A,op.op_linear,op.cache_linear.x)
end

function Algebra.allocate_residual(
  op::LinNonlinParamOperator,
  x::AbstractVector
  )

  b_lin = get_vector(op.cache_linear)
  similar(b_lin)
end

function Algebra.allocate_jacobian(
  op::LinNonlinParamOperator,
  x::AbstractVector
  )

  A_lin = get_matrix(op.cache_linear)
  similar(A_lin)
end

function Algebra.residual!(
  b::AbstractVector,
  op::LinNonlinParamOperator,
  x::AbstractVector
  )

  b_lin = get_vector(op.cache_linear)
  A_lin = get_matrix(op.cache_linear)
  residual!(b,op.op_nonlinear,x)
  mul!(b,A_lin,x,1,1)
  axpy!(1,b_lin,b)
  b
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  op::LinNonlinParamOperator,
  x::AbstractVector
  )

  A_lin = get_matrix(op.cache_linear)
  jacobian!(A,op.op_nonlinear,x)
  axpy!(1,A_lin,A)
  A
end

function Algebra.zero_initial_guess(op::LinNonlinParamOperator)
  v = get_vector(op.cache_linear)
  copy(v)
end