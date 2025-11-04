"""
    struct ParamStageOperator{O} <: NonlinearParamOperator
      op::ODEParamOperator{O}
      r::TransientRealization
      state_update::Function
      ws::Tuple{Vararg{Real}}
      paramcache::AbstractParamCache
    end

Stage operator to solve a parametric ODE with a time marching scheme
"""
struct ParamStageOperator{O} <: NonlinearParamOperator
  op::ODEParamOperator{O}
  r::TransientRealization
  state_update::Function
  ws::Tuple{Vararg{Real}}
  paramcache::AbstractParamCache
end

function Algebra.allocate_residual(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.state_update(x)
  allocate_residual(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.residual!(
  b::AbstractVector,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.state_update(x)
  residual!(b,nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.allocate_jacobian(
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.state_update(x)
  allocate_jacobian(nlop.op,nlop.r,usx,nlop.paramcache)
end

function Algebra.jacobian!(
  A::AbstractMatrix,
  nlop::ParamStageOperator,
  x::AbstractVector)

  usx = nlop.state_update(x)
  jacobian!(A,nlop.op,nlop.r,usx,nlop.ws,nlop.paramcache)
  A
end

function Algebra.solve(
  x::AbstractVector,
  ls::LinearSolver,
  op::ParamStageOperator,
  cache::Nothing)

  msg = """ Must preallocate a cache at the beginning of the time marching scheme.
  Check the Base.iterate function for a ODEParamSolution
  """
  error(msg)
end
