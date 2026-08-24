abstract type AssembleOperator end

function assemble_operator(a::AssembleOperator,U::FESpace,V::FESpace)
  @abstractmethod
end

function assemble_operator(a::AssembleOperator,feop::ParamFEOperator)
  assemble_operator(a,get_trial(feop),get_test(feop))
end

abstract type NormStyle <: AssembleOperator end

struct ℓ2 <: NormStyle end
struct L2 <: NormStyle end
struct H1 <: NormStyle end

const EuclideanNorm = ℓ2

abstract type CouplingStyle <: AssembleOperator end

struct DivCoupling <: CouplingStyle end

struct BlockOperator{A<:Tuple{Vararg{AssembleOperator}}} <: AssembleOperator
  op::A
end

function assemble_operator(::L2,U::SingleFieldFESpace,V::SingleFieldFESpace)
  l2_norm(U,V)
end

function assemble_operator(::H1,U::SingleFieldFESpace,V::SingleFieldFESpace)
  h1_norm(U,V)
end

function assemble_operator(op::NormStyle,X::MultiFieldFESpace,Y::MultiFieldFESpace)
  @check length(X) == length(Y) "MultiFieldFESpaces must have the same number of fields"
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)}()))
  assemble_operator(bop,X,Y)
end

function assemble_operator(op::BlockOperator{<:Tuple{Vararg{NormStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  map(assemble_operator,op.op,X.spaces,Y.spaces) |> _energy_mortar
end

function assemble_operator(op::BlockOperator{<:Tuple{Vararg{CouplingStyle}}},X::MultiFieldFESpace,Y::MultiFieldFESpace)
  V, = Y 
  map((o,U) -> assemble_operator(o,U,V),op.op,X.spaces) |> _coupling_mortar
end

for (f,g) in zip((:l2_norm,:h1_norm,:div_coupling),(:get_l2_form,:get_h1_form,:get_div_coupling_form))
  @eval begin
    $f(U::SingleFieldFESpace,V::SingleFieldFESpace) = assemble_matrix($g(U,V),U,V)
  end
end

function l2_norm(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  tproduct_array(mass_1d)
end

function h1_norm(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  stiff_1d = map(_stiffness_1d,U.spaces_1d,V.spaces_1d)
  decompositions = _find_decompositions(+,mass_1d,stiff_1d)
  GenericRankTensor(decompositions)
end

function div_coupling(U::TProductFESpace,V::TProductFESpace)
  mass_1d = map(_mass_1d,U.spaces_1d,V.spaces_1d)
  deriv_1d = map(_deriv_1d,U.spaces_1d,V.spaces_1d)
  decompositions = _find_decompositions(nothing,mass_1d,deriv_1d)
  GenericRankTensor(decompositions)
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

# _pad_to_rank(a::GenericRankTensor,K::Integer) = a

# function _pad_to_rank(a::Rank1Tensor,K::Integer)
#   K == 1 && return a
#   zero_decomp = Rank1Tensor(zero.(get_factors(a)))
#   GenericRankTensor(vcat(a,fill(zero_decomp,K-1)))
# end

# function _block_diag_tp_norm(norms::AbstractVector{<:Function},spaces::AbstractVector{<:TProductFESpace})
#   nfields = length(spaces)
#   diag = map((f,V) -> f(V),norms,spaces)
#   K = maximum(rank,diag)
#   diag = map(a -> _pad_to_rank(a,K),diag)
#   A = typeof(first(diag))
#   arrays = Array{A,2}(undef,nfields,nfields)
#   for j in 1:nfields, i in 1:nfields
#     arrays[i,j] = i == j ? diag[i] : _zero_rank_tensor(spaces[i],spaces[j],K)
#   end
#   BlockRankTensor(arrays)
# end

# _get_form(::typeof(l2_norm),V::SingleFieldFESpace) = get_l2_form(V)
# _get_form(::typeof(h1_norm),V::SingleFieldFESpace) = get_h1_form(V)

# function _dense_energy_form(norms::AbstractVector{<:Function},spaces::AbstractVector{<:SingleFieldFESpace})
#   forms = map(_get_form,norms,spaces)
#   function combined(x,y)
#     r = forms[1](x[1],y[1])
#     for i in 2:length(forms)
#       r += forms[i](x[i],y[i])
#     end
#     return r
#   end
#   return combined
# end

# function energy_norm(X::MultiFieldFESpace,norms::AbstractVector{<:Function})
#   spaces = X.spaces
#   @check length(norms) == length(spaces)
#   if all(V -> V isa TProductFESpace,spaces)
#     _block_diag_tp_norm(norms,spaces)
#   else
#     assemble_mat(_dense_energy_form(norms,spaces),X)
#   end
# end

# l2_norm(X::MultiFieldFESpace) = energy_norm(X,fill(l2_norm,length(X.spaces)))
# h1_norm(X::MultiFieldFESpace) = energy_norm(X,fill(h1_norm,length(X.spaces)))

# function _coupling(Xs::AbstractVector{<:TProductFESpace},Ys::AbstractVector{<:TProductFESpace})
#   nfields = length(Xs)
#   V = Xs[1]
#   D = length(V.spaces_1d)
#   primal_dual = map((U,Q) -> coupling(U,Q),Xs[2:end],Ys[2:end])
#   A = typeof(first(primal_dual))
#   arrays = Array{A,2}(undef,nfields,nfields)
#   for j in 1:nfields, i in 1:nfields
#     arrays[i,j] = (i == 1 && j > 1) ? primal_dual[j-1] : _zero_rank_tensor(Xs[i],Ys[j],D)
#   end
#   BlockRankTensor(arrays)
# end

# utils 

_energy_mortar(a) = mortar(a) 
_coupling_mortar(a) = mortar(a) 

function _energy_mortar(a::AbstractArray{<:AbstractRankTensor})
  if all(x -> x isa Rank1Tensor, a)
    return GenericRankTensor(a)
  else
    return mortar(a)
  end
end

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
  assemble_matrix((u,v) -> ∫(u*(∇(v)⋅v̂))dΩ,V,U)
end

function _zero_rank_tensor(U::TProductFESpace,V::TProductFESpace,K::Integer)
  factors = map(U.spaces_1d,V.spaces_1d) do Uid,Vjd
    spzeros(Float64,num_free_dofs(Uid),num_free_dofs(Vjd))
  end
  K == 1 ? Rank1Tensor(factors) : GenericRankTensor([Rank1Tensor(factors) for _ in 1:K])
end
