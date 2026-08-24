abstract type AssembleOperator end

function assemble_operator(a::AssembleOperator,U::FESpace,V::FESpace)
  @abstractmethod
end

function assemble_operator(a::AssembleOperator,feop::Union{ParamFEOperator,ParamOperator})
  assemble_operator(a,get_trial(feop),get_test(feop))
end

function assemble_operator(a::AssembleOperator,U::DirectSumFESpace,V::DirectSumFESpace)
  assemble_operator(a,get_bg_space(U),get_bg_space(V))
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
      γd::Float64
      hd::Float64
    end

H1 norm with an added Nitsche boundary penalty term, for spaces with weakly
imposed Dirichlet BCs (e.g. on a cut/aggregated embedded mesh):

`∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ + ∫((γd/hd)*v⋅u)dΓ`

where `dΓ` is built from `trian`.
"""
struct NitscheH1 <: NormStyle
  trian::Triangulation
  γd::Float64
  hd::Float64
end

function assemble_operator(op::NitscheH1,U::SingleFieldFESpace,V::SingleFieldFESpace)
  Uu,Vv = _unwrap_space(U),_unwrap_space(V)
  h1form = get_h1_form(Uu,Vv)
  degree = 2*max(get_polynomial_order(Uu),get_polynomial_order(Vv))
  dΓ = Measure(op.trian,degree)
  γdhd = op.γd/op.hd
  form(u,v) = h1form(u,v) + ∫(γdhd*(v⋅u))dΓ
  assemble_matrix(form,U,V)
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

# `get_trial(feop)` is typically an `UnEvalTrialFESpace` (aliased
# `ParamTrialFESpace`/`TransientTrialParamFESpace`) wrapping the actual space
# (e.g. a `TProductFESpace`) in its `.space` field. Unwrap it so that the
# `TProductFESpace`-specific l2_norm/h1_norm/div_coupling dispatches (which
# require an exact type match) are not silently skipped in favor of the
# generic dense fallback.
_unwrap_space(f::SingleFieldFESpace) = f
_unwrap_space(f::UnEvalTrialFESpace) = _unwrap_space(f.space)

function assemble_operator(::L2,U::SingleFieldFESpace,V::SingleFieldFESpace)
  l2_norm(_unwrap_space(U),_unwrap_space(V))
end

function assemble_operator(::H1,U::SingleFieldFESpace,V::SingleFieldFESpace)
  h1_norm(_unwrap_space(U),_unwrap_space(V))
end

function assemble_operator(::DivCoupling,U::SingleFieldFESpace,V::SingleFieldFESpace)
  div_coupling(_unwrap_space(U),_unwrap_space(V))
end

function assemble_operator(op::NormStyle,X::MultiFieldFESpace,Y::MultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)}()))
  assemble_operator(bop,X,Y)
end

function assemble_operator(op::CouplingStyle,X::MultiFieldFESpace,Y::MultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)-1}()))
  assemble_operator(bop,X,Y)
end

function assemble_operator(op::BlockOperator{<:Tuple{Vararg{NormStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  @check length(op) == length(X) == length(Y) "Wrong length of norms or MultiFieldFESpaces"
  map(assemble_operator,op.op,X.spaces,Y.spaces) |> _energy_mortar
end

function assemble_operator(op::BlockOperator{<:Tuple{Vararg{CouplingStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  @check length(op)+1 == length(X) == length(Y) "Wrong length of couplings or MultiFieldFESpaces"
  V, = Y.spaces
  Us = X.spaces[2:end]
  map((o,U) -> assemble_operator(o,U,V),op.op,Us) |> _coupling_mortar
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
  seminorm = TProduct._find_decompositions(mass_1d,stiff_1d)
  GenericRankTensor(vcat(Rank1Tensor(mass_1d),seminorm))
end

function div_coupling(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  deriv_1d = map(_deriv_1d,U.spaces_1d,V.spaces_1d)
  decompositions = TProduct._find_decompositions(mass_1d,deriv_1d)
  GenericRankTensor(decompositions)
end

# utils

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

_row_sizes(a::AbstractRankTensor{D}) where D = ntuple(d -> size(get_factor(a,d,1),1),Val{D}())
_col_sizes(a::AbstractRankTensor{D}) where D = ntuple(d -> size(get_factor(a,d,1),2),Val{D}())

function _zero_rank_tensor(row_sizes::NTuple{D,Int},col_sizes::NTuple{D,Int},K::Integer) where D
  factors = collect(map((nr,nc) -> spzeros(Float64,nr,nc),row_sizes,col_sizes))
  K == 1 ? Rank1Tensor(factors) : GenericRankTensor([Rank1Tensor(factors) for _ in 1:K])
end

# pads a Rank1Tensor/GenericRankTensor up to rank K with zero-valued
# decompositions, so that heterogeneous per-field norms (e.g. h1_norm on one
# field, l2_norm on another) can share a single concrete AbstractRankTensor
# type across all the blocks of one BlockRankTensor
_pad_to_rank(a::GenericRankTensor,K::Integer) = a
function _pad_to_rank(a::Rank1Tensor,K::Integer)
  K == 1 && return a
  zero_decomp = Rank1Tensor(zero.(get_factors(a)))
  GenericRankTensor(vcat(a,fill(zero_decomp,K-1)))
end

function _energy_mortar(a::AbstractVector{<:AbstractSparseMatrix})
  nfields = length(a)
  T = typeof(first(a))
  blocks = Matrix{T}(undef,nfields,nfields)
  for j in 1:nfields, i in 1:nfields
    blocks[i,j] = i == j ? a[i] : spzeros(size(a[i],1),size(a[j],1))
  end
  mortar(blocks)
end

function _energy_mortar(a::AbstractVector{<:AbstractRankTensor})
  nfields = length(a)
  K = maximum(rank,a)
  diag = map(x -> _pad_to_rank(x,K),a)
  A = typeof(first(diag))
  blocks = Matrix{A}(undef,nfields,nfields)
  for j in 1:nfields, i in 1:nfields
    blocks[i,j] = i == j ? diag[i] : _zero_rank_tensor(_row_sizes(diag[i]),_row_sizes(diag[j]),K)
  end
  BlockRankTensor(blocks)
end

function _coupling_mortar(a::AbstractVector{<:AbstractSparseMatrix})
  ndual = length(a)
  nfields = ndual+1
  nprimal = size(first(a),1)
  ncols = map(x -> size(x,2),a)
  T = typeof(first(a))
  blocks = Matrix{T}(undef,nfields,nfields)
  blocks[1,1] = spzeros(nprimal,nprimal)
  for i in 1:ndual
    blocks[1,i+1] = a[i]
    blocks[i+1,1] = spzeros(ncols[i],nprimal)
    for j in 1:ndual
      blocks[i+1,j+1] = spzeros(ncols[i],ncols[j])
    end
  end
  mortar(blocks)
end

function _coupling_mortar(a::AbstractVector{<:AbstractRankTensor})
  ndual = length(a)
  nfields = ndual+1
  K = maximum(rank,a)
  a = map(x -> _pad_to_rank(x,K),a)
  A = typeof(first(a))
  primal_sizes = _row_sizes(first(a))
  dual_sizes = map(_col_sizes,a)
  blocks = Matrix{A}(undef,nfields,nfields)
  blocks[1,1] = _zero_rank_tensor(primal_sizes,primal_sizes,K)
  for i in 1:ndual
    blocks[1,i+1] = a[i]
    blocks[i+1,1] = _zero_rank_tensor(dual_sizes[i],primal_sizes,K)
    for j in 1:ndual
      blocks[i+1,j+1] = _zero_rank_tensor(dual_sizes[i],dual_sizes[j],K)
    end
  end
  BlockRankTensor(blocks)
end
