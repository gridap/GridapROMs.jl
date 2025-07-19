"""
"""
abstract type FEDomainOperator{O<:OperatorType,T<:TriangulationStyle} <: FEOperator end

"""
"""
const JointFEOperator{O<:OperatorType} = FEDomainOperator{O,JointDomains}

"""
"""
const SplitFEOperator{O<:OperatorType} = FEDomainOperator{O,SplitDomains}

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
  DomainOperator(feop)
end

CellData.get_domains(feop::FEDomainOperator) = @abstractmethod
get_domains_res(feop::FEDomainOperator) = get_domains_res(get_domains(feop))
get_domains_jac(feop::FEDomainOperator) = get_domains_jac(get_domains(feop))

struct GenericFEDomainOperator{O<:OperatorType,T<:TriangulationStyle} <: FEDomainOperator{O,T}
  feop::FEOperator
  domains::FEDomains
end

get_fe_operator(feop::GenericFEDomainOperator) = feop.feop
CellData.get_domains(feop::GenericFEDomainOperator) = feop.domains

function LinearFEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace)
  jac′(u,du,v) = jac(du,v)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res,jac′,trial,test,assem)
  domains = FEDomains()
  GenericFEDomainOperator{LinearEq,JointDomains}(feop,domains)
end

function LinearFEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,domains::FEDomains)
  jac′(u,du,v,args...) = jac(du,v,args...)
  res′,jac′′ = _set_domains(res,jac′,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res′,jac′′,trial,test,assem)
  GenericFEDomainOperator{LinearEq,SplitDomains}(feop,domains)
end

function FESpaces.FEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res,jac,trial,test,assem)
  domains = FEDomains()
  GenericFEDomainOperator{NonlinearEq,JointDomains}(feop,domains)
end

function FESpaces.FEOperator(res::Function,jac::Function,trial::FESpace,test::FESpace,domains::FEDomains)
  res′,jac′ = _set_domains(res,jac,test,trial,domains)
  assem = SparseMatrixAssembler(trial,test)
  feop = FEOperator(res′,jac′,trial,test,assem)
  GenericFEDomainOperator{NonlinearEq,SplitDomains}(feop,domains)
end

for f in (:LinearFEOperator,:(FESpaces.FEOperator))
  @eval begin
    function $f(res::Function,jac::Function,trial::FESpace,test::FESpace,trians...)
      domains = FEDomains(trians...)
      $f(res,jac,trial,test,domains)
    end
  end
end

# triangulation utils

for f in (:set_domains,:change_domains)
  T = f == :set_domains ? :JointDomains : :SplitDomains
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
      GenericFEDomainOperator{O,$T}(feop,domains′)
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
