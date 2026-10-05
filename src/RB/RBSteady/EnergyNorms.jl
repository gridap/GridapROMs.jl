abstract type AssembleOperator end

function _assemble_operator(a::AssembleOperator,U::FESpace,V::FESpace)
  @abstractmethod
end

function _assemble_operator(a::AssembleOperator,U::DirectSumFESpace,V::DirectSumFESpace)
  _assemble_operator(a,get_bg_space(U),get_bg_space(V))
end

function assemble_operator(a::AssembleOperator,feop)
  _assemble_operator(a,_unwrap(get_trial(feop)),get_test(feop))
end

"""
    abstract type NormStyle <: AssembleOperator end

Subtypes:
- [`ℓ2`](@ref) (aliased [`EuclideanNorm`](@ref))
- [`L2`](@ref)
- [`H1`](@ref)
- [`NitscheH1`](@ref)
"""
abstract type NormStyle <: AssembleOperator end

struct ℓ2 <: NormStyle end
struct L2 <: NormStyle end
struct H1 <: NormStyle end

"""
    const EuclideanNorm = ℓ2
"""
const EuclideanNorm = ℓ2

"""
    struct NitscheH1 <: NormStyle
      trian::Triangulation
      γ::Float64
      h::Float64
    end

H1 norm with an added Nitsche boundary penalty term, for spaces with weakly
imposed Dirichlet BCs (e.g. on a cut/aggregated embedded mesh):

`∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ + ∫((γ/h)*v⋅u)dΓ`

where `dΓ` is built from `trian`.
"""
struct NitscheH1 <: NormStyle
  trian::Triangulation
  γ::Float64
  h::Float64
end

function _assemble_operator(op::NitscheH1,U::SingleFieldFESpace,V::SingleFieldFESpace)
  h1form = get_h1_form(U,V)
  degree = 2*max(get_polynomial_order(U),get_polynomial_order(V))
  dΓ = Measure(op.trian,degree)
  form(u,v) = h1form(u,v) + ∫((op.γ/op.h)*(v⋅u))dΓ
  assemble_matrix(form,U,V)
end

struct EnergyNorm{F<:Function} <: NormStyle
  form::F
end

for T in (:SingleFieldFESpace,:MultiFieldFESpace)
  @eval begin
    function _assemble_operator(op::EnergyNorm,U::$T,V::$T) 
      assemble_matrix(op.form,U,V)
    end
  end
end

"""
    abstract type CouplingStyle <: AssembleOperator end

Subtypes:
- [`DivCoupling`](@ref)
"""
abstract type CouplingStyle <: AssembleOperator end

"""
    struct DivCoupling <: CouplingStyle end

Divergence coupling between a (vector-valued) primal field `v` and a dual
field `p`, i.e. the bilinear form `∫(p*(∇⋅v))dΩ`.
"""
struct DivCoupling <: CouplingStyle end

"""
    struct BlockOperator{A<:Tuple{Vararg{AssembleOperator}}} <: AssembleOperator
      op::A
    end

Per-field [`AssembleOperator`](@ref), for a `MultiFieldFESpace`. E.g., a
Stokes-like energy norm (H1 for velocity, L2 for pressure) is
`BlockOperator((H1(),L2()))`; a divergence coupling for two dual fields is
`BlockOperator((DivCoupling(),DivCoupling()))`.
"""
struct BlockOperator{A<:Tuple{Vararg{AssembleOperator}}} <: AssembleOperator
  op::A
end

Base.length(op::BlockOperator) = length(op.op)

"""
    const BlockNorm = BlockOperator
"""
const BlockNorm = BlockOperator

"""
    const BlockCoupling = BlockOperator
"""
const BlockCoupling = BlockOperator

function _assemble_operator(::L2,U::SingleFieldFESpace,V::SingleFieldFESpace)
  l2_norm(U,V)
end

function _assemble_operator(::H1,U::SingleFieldFESpace,V::SingleFieldFESpace)
  h1_norm(U,V)
end

function _assemble_operator(::DivCoupling,U::SingleFieldFESpace,V::SingleFieldFESpace)
  div_coupling(U,V)
end

get_form(::L2,U::FESpace,V::FESpace) = get_l2_form(U,V)
get_form(::H1,U::FESpace,V::FESpace) = get_h1_form(U,V)
get_form(op::EnergyNorm,U::FESpace,V::FESpace) = op.form
get_form(::DivCoupling,U::FESpace,V::FESpace) = get_div_coupling_form(U,V)

function _assemble_operator(op::NormStyle,X::MultiFieldFESpace,Y::MultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)}()))
  _assemble_operator(bop,X,Y)
end

function _assemble_operator(op::CouplingStyle,X::MultiFieldFESpace,Y::MultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)-1}()))
  _assemble_operator(bop,X,Y)
end

function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{NormStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  @check length(op) == length(X) == length(Y) "Wrong length of norms or MultiFieldFESpaces"
  map(_assemble_operator,op.op,X.spaces,Y.spaces)
end

function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{CouplingStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  @check length(op)+1 == length(X) == length(Y) "Wrong length of couplings or MultiFieldFESpaces"
  V, = Y.spaces
  Us = X.spaces[2:end]
  map((o,U) -> _assemble_operator(o,U,V),op.op,Us)
end

for (f,g) in zip((:l2_norm,:h1_norm,:div_coupling),(:get_l2_form,:get_h1_form,:get_div_coupling_form))
  @eval $f(U::SingleFieldFESpace,V::SingleFieldFESpace) = assemble_matrix($g(U,V),U,V)
end

function get_l2_form(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  return (u,v) -> ∫(v⋅u)dΩ
end

function get_h1_form(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  return (u,v) -> ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ
end

function get_div_coupling_form(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  return (p,v) -> ∫(p*(∇⋅v))dΩ
end

function l2_norm(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  Rank1Tensor(mass_1d)
end

function h1_norm(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  stiff_1d = map(_stiffness_1d,U.spaces_1d,V.spaces_1d)
  inds = LinearIndices(mass_1d)
  map(inds) do i
    di = copy(mass_1d)
    di[i] += stiff_1d[i]
    Rank1Tensor(di)
  end |> GenericRankTensor
end

function div_coupling(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  deriv_1d = map(_deriv_1d,U.spaces_1d,V.spaces_1d)
  inds = LinearIndices(mass_1d)
  map(inds) do i
    di = copy(mass_1d)
    di[i] = deriv_1d[i]
    Rank1Tensor(di)
  end |> GenericRankTensor
end

# utils

_unwrap(f) = f 
_unwrap(f::UnEvalTrialFESpace) = _unwrap(f.space)
_unwrap(f::MultiFieldFESpace) = MultiFieldFESpace(map(_unwrap,f.spaces);style=MultiFieldStyle(f))

function _meas(V::FESpace,Q::FESpace)
  Ωv = get_triangulation(V)
  Ωq = get_triangulation(Q)
  @check Ωv === Ωq "FESpaces must share the same triangulation"
  orderv = get_polynomial_order(V)
  orderq = get_polynomial_order(Q)
  order = max(orderv,orderq)
  Measure(Ωv,2*order)
end

function _mass_1d(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  assemble_matrix((u,v) -> ∫(u*v)dΩ,U,V)
end

function _stiffness_1d(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  assemble_matrix((u,v) -> ∫(∇(u)⋅∇(v))dΩ,U,V)
end

function _deriv_1d(U::SingleFieldFESpace,V::SingleFieldFESpace)
  dΩ = _meas(U,V)
  v̂ = VectorValue(1.0)
  assemble_matrix((u,v) -> ∫(u*(∇(v)⋅v̂))dΩ,U,V)
end