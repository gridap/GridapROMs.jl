function _assemble_operator(op::NitscheH1,U::DistributedSingleFieldFESpace,V::DistributedSingleFieldFESpace)
  h1form = get_h1_form(U,V)
  degree = 2*max(get_polynomial_order(U),get_polynomial_order(V))
  dΓ = Measure(op.trian,degree)
  form(u,v) = h1form(u,v) + ∫((op.γ/op.h)*(v⋅u))dΓ
  assemble_matrix(form,U,V)
end

for T in (:DistributedSingleFieldFESpace,:DistributedMultiFieldFESpace)
  @eval begin
    function _assemble_operator(op::EnergyNorm,U::$T,V::$T) 
      assemble_matrix(op.form,U,V)
    end
  end
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

# `_energy_mortar`/`_coupling_mortar` (the serial-only "assemble every field's
# operator separately, then hand-mortar a block matrix padded with `spzeros`
# off-diagonal blocks") have no distributed analogue: a "zero" `PSparseMatrix`
# block needs a matching row/col `PRange`, and GridapDistributed has no ready
# constructor for that. Instead, build ONE combined multi-field bilinear form
# (block-diagonal for norms, block-arrow for couplings - see `get_form`) and
# assemble it in a single `assemble_matrix` call: GridapDistributed's own
# multi-field assembler then derives exactly the same block sparsity as the
# serial `_energy_mortar`/`_coupling_mortar` construction, with every
# untouched block implicitly (and correctly) empty.
function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{NormStyle}}},X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  @check length(op) == length(X) == length(Y) "Wrong length of norms or MultiFieldFESpaces"
  forms = map(get_form,op.op,X.field_fe_space,Y.field_fe_space)
  form(u,v) = sum(forms[i](u[i],v[i]) for i in eachindex(forms))
  assemble_matrix(form,X,Y)
end

function _assemble_operator(op::BlockOperator{<:Tuple{Vararg{CouplingStyle}}},X::DistributedMultiFieldFESpace,Y::DistributedMultiFieldFESpace)
  @check length(op)+1 == length(X) == length(Y) "Wrong length of couplings or MultiFieldFESpaces"
  Us = X.field_fe_space[2:end]
  forms = map((o,U) -> get_form(o,U,Y.field_fe_space[1]),op.op,Us)
  form(u,v) = sum(forms[i](u[i+1],v[1]) for i in eachindex(forms))
  assemble_matrix(form,X,Y)
end

function _unwrap(f::DistributedMultiFieldFESpace)
  DistributedMultiFieldFESpace(
    f.field_fe_space,
    map(_unwrap,local_views(f)),
    f.gids,
    f.vector_type
  )
end