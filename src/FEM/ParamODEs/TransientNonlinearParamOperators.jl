function Algebra.allocate_residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function Algebra.residual!(
  b,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache;
  add::Bool=false)

  @abstractmethod
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}})

  paramcache = allocate_paramcache(nlop,r;evaluated=true)
  residual(nlop,r,us,paramcache)
end

function Algebra.residual(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  b = allocate_residual(nlop,r,us,paramcache)
  residual!(b,nlop,r,us,paramcache)
  b
end

function Algebra.allocate_jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  paramcache)

  @abstractmethod
end

function ODEs.jacobian_add!(
  A,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  @abstractmethod
end

function Algebra.jacobian!(
  A,
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  LinearAlgebra.fillstored!(A,zero(eltype(A)))
  jacobian_add!(A,nlop,r,us,ws,paramcache)
  A
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}})

  paramcache = allocate_paramcache(nlop,r;evaluated=true)
  jacobian(nlop,r,us,ws,paramcache)
end

function Algebra.jacobian(
  nlop::NonlinearParamOperator,
  r::TransientRealization,
  us::Tuple{Vararg{AbstractVector}},
  ws::Tuple{Vararg{Real}},
  paramcache)

  A = allocate_jacobian(nlop,r,us,paramcache)
  jacobian!(A,nlop,r,us,ws,paramcache)
  A
end
