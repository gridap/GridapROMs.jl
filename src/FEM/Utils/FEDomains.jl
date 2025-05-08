filter_domains(a) = a
filter_domains(a::Tuple) = map(filter_domains,a)

"""
    struct FEDomains{A,B}
      domains_res::A
      domains_jac::B
    end

Fields:
- `domains_res`: triangulations relative to the residual (nothing by default)
- `domains_jac`: triangulations relative to the Jacobian (nothing by default)
"""
struct FEDomains{A,B}
  domains_res::A
  domains_jac::B

  function FEDomains(domains_res,domains_jac)
    domains_res′ = filter_domains(domains_res)
    domains_jac′ = filter_domains(domains_jac)
    A = typeof(domains_res′)
    B = typeof(domains_jac′)
    new{A,B}(domains_res′,domains_jac′)
  end
end

FEDomains(args...) = FEDomains(nothing,nothing)

get_domains_res(d::FEDomains) = d.domains_res
get_domains_jac(d::FEDomains) = d.domains_jac

abstract type OperatorType <: GridapType end
struct LinearEq <: OperatorType end
struct NonlinearEq <: OperatorType end

abstract type FEDomainOperator{O<:OperatorType} <: FEOperator end

"""
    get_fe_operator(op::ParamOperator) -> ParamFEOperator

Fetches the underlying FE operator of an algebraic operator `op`
"""
get_fe_operator(feop::FEDomainOperator) = @abstractmethod

FESpaces.get_test(feop::FEDomainOperator) = get_test(get_fe_operator(feop))
FESpaces.get_trial(feop::FEDomainOperator) = get_trial(get_fe_operator(feop))
ODEs.get_res(feop::FEDomainOperator) = get_res(get_fe_operator(feop))
get_jac(feop::FEDomainOperator) = get_jac(get_fe_operator(feop))
ODEs.get_assembler(feop::FEDomainOperator) = get_assembler(get_fe_operator(feop))

function FESpaces.get_algebraic_operator(feop::FEDomainOperator)
  GenericDomainOperator(feop)
end

CellData.get_domains(feop::FEDomainOperator) = @abstractmethod
get_domains_res(feop::FEDomainOperator) = get_domains_res(get_domains(feop))
get_domains_jac(feop::FEDomainOperator) = get_domains_jac(get_domains(feop))

function Algebra.allocate_residual(feop::FEDomainOperator,uh)
  allocate_residual(get_fe_operator(feop),uh)
end

function Algebra.residual!(b::AbstractVector,feop::FEDomainOperator,uh)
  residual!(b,get_fe_operator(feop),uh)
end

function Algebra.allocate_jacobian(feop::FEDomainOperator,uh)
  allocate_jacobian(get_fe_operator(feop),uh)
end

function Algebra.jacobian!(A::AbstractMatrix,feop::FEDomainOperator,uh)
  jacobian!(A,get_fe_operator(feop),uh)
end

function Algebra.residual_and_jacobian!(
  b::AbstractVector,A::AbstractMatrix,feop::FEDomainOperator,uh)
  residual_and_jacobian!(b,A,get_fe_operator(feop),uh)
end

function Algebra.residual_and_jacobian(feop::FEDomainOperator,uh)
  residual_and_jacobian!(get_fe_operator(feop),uh)
end

struct GenericFEDomainOperator{O<:OperatorType} <: FEDomainOperator{O}
  feop::FEOperator
  domains::FEDomains
end

get_fe_operator(feop::GenericFEDomainOperator) = feop.feop
CellData.get_domains(feop::GenericFEDomainOperator) = feop.domains

function LinearFEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,domains::FEDomains)
  jac′(u,du,v,args...) = jac(du,v,args...)
  res′,jac′′ = _set_domains(res,jac′,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res′,jac′′,trial,test,assem)
  GenericFEDomainOperator{LinearEq}(feop,domains)
end

function LinearFEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,trians...)
  LinearFEOperator(res,jac,trial,test,FEDomains(trians...))
end

function FESpaces.FEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,domains::FEDomains)
  res′,jac′ = _set_domains(res,jac,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res′,jac′,trial,test,assem)
  GenericFEDomainOperator{NonlinearEq}(feop,domains)
end

function FESpaces.FEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,trians...)
  FEOperator(res,jac,trial,test,FEDomains(trians...))
end

abstract type NonlinearDomainOperator{O<:OperatorType} <: NonlinearOperator end

get_fe_operator(op::NonlinearDomainOperator) = @abstractmethod
FESpaces.get_test(op::NonlinearDomainOperator) = get_test(get_fe_operator(op))
FESpaces.get_trial(op::NonlinearDomainOperator) = get_trial(get_fe_operator(op))
ODEs.get_res(op::NonlinearDomainOperator) = get_res(get_fe_operator(op))
get_jac(op::NonlinearDomainOperator) = get_jac(get_fe_operator(op))
ODEs.get_assembler(op::NonlinearDomainOperator) = get_assembler(get_fe_operator(op))

CellData.get_domains(op::NonlinearDomainOperator) = get_domains(get_fe_operator(op))
get_domains_res(op::NonlinearDomainOperator) = get_domains_res(get_domains(op))
get_domains_jac(op::NonlinearDomainOperator) = get_domains_jac(get_domains(op))

function Algebra.allocate_residual(op::NonlinearDomainOperator,u::AbstractVector)
  trial = get_trial(op)
  uh = EvaluationFunction(trial,u)
  allocate_residual(get_fe_operator(op),uh)
end

function Algebra.residual!(b::AbstractVector,op::NonlinearDomainOperator,u::AbstractVector)
  trial = get_trial(op)
  uh = EvaluationFunction(trial,u)
  residual!(b,get_fe_operator(op),uh)
end

function Algebra.residual!(b::AbstractVector,op::NonlinearDomainOperator{LinearEq},u::AbstractVector)
  trial = get_trial(op)
  uh0 = zero(trial)
  residual!(b,get_fe_operator(op),uh0)
  rmul!(b,-1)
end

function Algebra.allocate_jacobian(op::NonlinearDomainOperator,u::AbstractVector)
  trial = get_trial(op)
  uh = EvaluationFunction(trial,u)
  allocate_jacobian(get_fe_operator(op),uh)
end

function Algebra.jacobian!(A::AbstractMatrix,op::NonlinearDomainOperator,u::AbstractVector)
  trial = get_trial(op)
  uh = EvaluationFunction(trial,u)
  jacobian!(A,get_fe_operator(op),uh)
end

function Algebra.zero_initial_guess(op::NonlinearDomainOperator)
  trial = get_trial(op)
  zero_free_values(trial)
end

set_domains(op::NonlinearDomainOperator,args...) = get_algebraic_operator(set_domains(get_fe_operator(op),args...))
change_domains(op::NonlinearDomainOperator,args...) = get_algebraic_operator(change_domains(get_fe_operator(op),args...))

struct GenericDomainOperator{O<:OperatorType} <: NonlinearDomainOperator{O}
  feop::FEDomainOperator{O}
end

get_fe_operator(op::GenericDomainOperator) = op.feop

function GenericDomainOperator(args...;kwargs...)
  feop = FEOperator(args...;kwargs...)
  get_algebraic_operator(feop)
end

# solve utils

function Algebra.solve!(
  x::AbstractVector,
  solver::LinearSolver,
  op::NonlinearDomainOperator{LinearEq}
  )

  u = zero_initial_guess(op)
  A = jacobian(op,u)
  b = residual(op,u)
  solve!(x,solver,A,b)
end

# triangulation utils

for f in (:set_domains,:change_domains)
  @eval begin
    function $f(feop::FEDomainOperator)
      $f(feop,get_domains(feop))
    end

    function $f(feop::FEDomainOperator,domains::FEDomains)
      $f(feop,get_domains_res(domains),get_domains_jac(domains))
    end

    function $f(feop::GenericFEDomainOperator{O},trians_res,trians_jac) where O
      trian_res′ = order_domains(get_domains_res(feop),trians_res)
      trian_jac′ = order_domains(get_domains_jac(feop),trians_jac)
      res′,jac′ = _set_domains(
        get_res(feop),get_jac(feop),get_trial(feop),get_test(feop),trian_res′,trian_jac′
      )
      domains′ = FEDomains(trian_res′,trian_jac′)
      feop = FEOperator(res′,jac′,get_trial(feop),get_test(feop),get_assembler(feop))
      GenericFEDomainOperator{O}(feop,domains′)
    end
  end
end

function _set_domains(
  res::Function,
  jac::Function,
  test::FESpace,
  trial::FESpace,
  trian_res::Tuple{Vararg{Triangulation}},
  trian_jac::Tuple{Vararg{Triangulation}})

  polyn_order = get_polynomial_order(test)
  @check polyn_order == get_polynomial_order(trial)
  res′ = _set_domain_res(res,trian_res,polyn_order)
  jac′ = _set_domain_jac(jac,trian_jac,polyn_order)
  return res′,jac′
end

function _set_domains(
  res::Function,
  jac::Function,
  test::FESpace,
  trial::FESpace,
  domains::FEDomains)

  trian_res = get_domains_res(domains)
  trian_jac = get_domains_jac(domains)
  _set_domains(res,jac,test,trial,trian_res,trian_jac)
end

function _set_domain_jac(
  jac::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  jac′(u,du,v,args...) = jac(u,du,v,args...)
  jac′(u,du,v) = jac′(u,du,v,meas...)
  return jac′
end

function _set_domain_res(
  res::Function,
  trian::Tuple{Vararg{Triangulation}},
  order)

  degree = 2*order
  meas = Measure.(trian,degree)
  res′(u,v,args...) = res(u,v,args...)
  res′(u,v) = res′(u,v,meas...)
  return res′
end

# utils

ODEs.get_res(feop::FESpaces.FEOperatorFromWeakForm) = feop.res
get_jac(feop::FESpaces.FEOperatorFromWeakForm) = feop.jac
ODEs.get_assembler(feop::FESpaces.FEOperatorFromWeakForm) = feop.assem

"""
    get_polynomial_order(f::FESpace) -> Integer

Retrieves the polynomial order of `f`
"""
get_polynomial_order(f::SingleFieldFESpace) = get_polynomial_order(get_fe_basis(f))
get_polynomial_order(f::MultiFieldFESpace) = maximum(map(get_polynomial_order,f.spaces))

function get_polynomial_order(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_order(shapefun.fields)
end

"""
    get_polynomial_orders(fs::FESpace) -> Integer

Retrieves the polynomial order of `fs` for every dimension
"""
get_polynomial_orders(fs::SingleFieldFESpace) = get_polynomial_orders(get_fe_basis(fs))
get_polynomial_orders(fs::MultiFieldFESpace) = maximum.(map(get_polynomial_orders,fs.spaces))

function get_polynomial_orders(basis)
  cell_basis = get_data(basis)
  shapefun = testitem(cell_basis)
  get_orders(shapefun.fields)
end
