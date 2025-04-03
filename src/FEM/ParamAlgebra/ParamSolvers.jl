# linear interface

function Algebra.solve(
  ls::LinearSolver,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

  x = allocate_in_domain(A)
  solve!(x,ls,A,b)
  x
end

function Algebra.solve!(
  x::AbstractParamVector,
  ls::LinearSolver,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

  A_item = testitem(A)
  x_item = testitem(x)
  ss = symbolic_setup(ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)
  solve!(x,ns,A,b)
end

function Algebra.solve!(
  x::AbstractParamVector,
  ns::NumericalSetup,
  A::AbstractParamMatrix,
  b::AbstractParamVector)

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

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::NonlinearParamOperator,
  cache::Nothing)

  cache = allocate_systemcache(op,x)
  solve!(x,ls,op,cache)
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::NonlinearParamOperator,
  cache::SystemCache)

  fill!(x,zero(eltype(x)))
  @unpack A,b = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  ns = solve!(x,ls,A,b)
  Algebra.LinearSolverCache(A,b,ns)
end

function Algebra.solve!(
  x::AbstractVector,
  ls::LinearSolver,
  op::NonlinearParamOperator,
  cache::Algebra.LinearSolverCache)

  fill!(x,zero(eltype(x)))
  @unpack A,b,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)
  solve!(x,ns,A,b)
  cache
end

# nonlinear interface

function Algebra.solve!(
  x::AbstractVector,
  nls::NewtonSolver,
  op::NonlinearParamOperator,
  cache::Nothing)

  cache = allocate_systemcache(op,x)
  solve!(x,ls,op,cache)
end

function Algebra.solve!(
  x::AbstractParamVector,
  nls::NewtonSolver,
  op::NonlinearParamOperator,
  cache::SystemCache)

  fill!(x,zero(eltype(x)))
  update_systemcache!(op,x)

  @unpack A,b = cache
  residual!(b,op,x)
  jacobian!(A,op,x)

  A_item = testitem(A)
  x_item = testitem(x)
  dx = allocate_in_domain(A_item)
  fill!(dx,zero(eltype(dx)))
  ss = symbolic_setup(nls.ls,A_item)
  ns = numerical_setup(ss,A_item,x_item)

  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  return NonlinearSolvers.NewtonCache(A,b,dx,ns)
end

function Algebra.solve!(
  x::AbstractVector,
  nls::NewtonSolver,
  op::NonlinearParamOperator,
  cache::NonlinearSolvers.NewtonCache)

  update_systemcache!(op,x)

  @unpack A,b,dx,ns = cache
  residual!(b,op,x)
  jacobian!(A,op,x)

  Algebra._solve_nr!(x,A,b,dx,ns,nls,op)
  return cache
end

function Algebra._solve_nr!(
  x::AbstractParamVector,
  A::AbstractParamMatrix,
  b::AbstractParamVector,
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
