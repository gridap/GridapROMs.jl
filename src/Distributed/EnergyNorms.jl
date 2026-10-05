function _assemble_operator(op::NitscheH1,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  h1form = get_h1_form(U,V)
  degree = 2*max(get_polynomial_order(U),get_polynomial_order(V))
  dΓ = Measure(op.trian,degree)
  form(u,v) = h1form(u,v) + ∫((op.γ/op.h)*(v⋅u))dΓ
  assemble_matrix(form,U,V)
end

function _assemble_operator(op::EnergyNorm,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  assemble_matrix(op.form,U,V)
end

# op.form is a single, user-provided function of every field jointly (unlike
# the BlockOperator path below, which assembles each field's own, separate
# operator), so it isn't generically splittable into per-field pieces here,
# and this (more specific than the NormStyle/BlockOperator path below)
# previously assembled it directly on U,V as given, producing a monolithic
# PSparseMatrix for the default (consecutive) DistributedMultiFieldFESpace
# numbering. U,V are re-styled to BlockMultiFieldStyle purely for this
# assembly (independent of whatever style the actual PDE solve uses U,V
# with), so assemble_matrix produces a genuine BlockPMatrix instead: op.form
# only ever sees trial/test basis functions, never the underlying
# MultiFieldStyle, so this is transparent to its result -- it just changes
# how the assembled matrix is stored (block vs monolithic), which is what
# the downstream per-field projection code (RBSteady.projection(red,s::
# BlockSnapshots,X), which extracts each field's diagonal block via
# X[Block(i,i)]) needs.
function _assemble_operator(op::EnergyNorm,U::DistributedMultiFieldFESpace,V::DistributedMultiFieldFESpace)
  Ub = MultiFieldFESpace(collect(U.field_fe_space);style=BlockMultiFieldStyle())
  Vb = MultiFieldFESpace(collect(V.field_fe_space);style=BlockMultiFieldStyle())
  assemble_matrix(op.form,Ub,Vb)
end

function _assemble_operator(::L2,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  l2_norm(U,V)
end

function _assemble_operator(::H1,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  h1_norm(U,V)
end

function _assemble_operator(::DivCoupling,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  div_coupling(U,V)
end

for (f,g) in zip((:l2_norm,:h1_norm,:div_coupling),(:get_l2_form,:get_h1_form,:get_div_coupling_form))
  @eval $f(U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace) = assemble_matrix($g(U,V),U,V)
end

function get_l2_form(U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  dΩ = _meas(U,V)
  return (u,v) -> ∫(v⋅u)dΩ
end

function get_h1_form(U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  dΩ = _meas(U,V)
  return (u,v) -> ∫(v⋅u)dΩ + ∫(∇(v)⊙∇(u))dΩ
end

function get_div_coupling_form(U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  dΩ = _meas(U,V)
  return (p,v) -> ∫(p*(∇⋅v))dΩ
end

function _assemble_operator(op::NormStyle,X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)}()))
  _assemble_operator(bop,X,Y)
end

function _assemble_operator(op::CouplingStyle,X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  bop = BlockOperator(ntuple(_ -> op,Val{length(X)-1}()))
  _assemble_operator(bop,X,Y)
end

function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{NormStyle}}},X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  @check length(op) == length(X) == length(Y) "Wrong length of norms or MultiFieldFESpaces"
  map(_assemble_operator,op.op,X.field_fe_space,Y.field_fe_space)
end

function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{CouplingStyle}}},X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  @check length(op)+1 == length(X) == length(Y) "Wrong length of couplings or MultiFieldFESpaces"
  V, = Y.field_fe_space
  Us = X.field_fe_space[2:end]
  map((o,U) -> _assemble_operator(o,U,V),op.op,Us)
end

function _unwrap(f::DistributedMultiFieldFESpace)
  DistributedMultiFieldFESpace(
    f.field_fe_space,
    map(_unwrap,local_views(f)),
    f.gids,
    f.vector_type
  )
end