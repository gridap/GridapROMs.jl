const AbstractParamPVector = Union{PVector{<:AbstractParamVector},BlockPArray{<:AbstractParamVector}}
const AbstractParamPSparseMatrix = Union{PSparseMatrix{<:ParamSparseMatrix},BlockPArray{<:ParamSparseMatrix}}

function Algebra.solve!(
  x::AbstractParamPVector,
  ls::LinearSolver,
  A::AbstractParamPSparseMatrix,
  b::AbstractParamPVector)

  A_item = param_getindex(A,1)
  x_item = param_getindex(x,1)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)
  solve!(x,ns,A,b)
end

function Algebra.solve!(
  x::AbstractParamPVector,
  ns::NumericalSetup,
  A::AbstractParamPSparseMatrix,
  b::AbstractParamPVector)

  @inbounds for i in param_eachindex(x)
    Ai = param_getindex(A,i)
    xi = param_getindex(x,i)
    bi = param_getindex(b,i)
    rmul!(bi,-1)
    numerical_setup!(ns,Ai)
    solve!(xi,ns,bi)
  end

  ns
end

function Algebra._solve_nr!(
  x::AbstractParamPVector,
  A::AbstractParamPSparseMatrix,
  b::AbstractParamPVector,
  dx,ns,nls,op)

  log = nls.log

  res = norm(b)
  done = LinearSolvers.init!(log,res)

  while !done
    @inbounds for i in param_eachindex(x)
      xi = param_getindex(x,i)
      Ai = param_getindex(A,i)
      bi = param_getindex(b,i)
      numerical_setup!(ns,Ai)
      rmul!(bi,-1)
      solve!(dx,ns,bi)
      xi .+= dx
    end

    residual!(b,op,x)
    res  = norm(b)
    done = LinearSolvers.update!(log,res)

    if !done
      jacobian!(A,op,x)
    end
  end

  LinearSolvers.finalize!(log,res)
  return x
end
