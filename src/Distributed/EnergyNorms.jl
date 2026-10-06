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

function _assemble_operator(op::EnergyNorm,U::DistributedMultiFieldFESpace,V::DistributedMultiFieldFESpace)
  A = assemble_matrix(op.form,U,V)
  if A isa BlockPArray
    # a BlockMultiFieldStyle U,V (e.g. after _setup/_convert_to_block)
    # assembles directly into a block-structured matrix, one block per
    # field pair; no manual row/col restriction needed, unlike the
    # ConsecutiveMultiFieldStyle case below (monolithic PSparseMatrix)
    map(i -> A[Block(i,i)],1:num_fields(U))
  else
    local_ranges = _get_local_ranges(map(Ui -> partition(get_free_dof_ids(Ui)),U.field_fe_space))
    map(lr -> _restrict_diag_block(A,lr),local_ranges)
  end
end

function _restrict_diag_block(A::PSparseMatrix,local_range)
  row_positions,new_row_partition = _restrict_to_local_range(partition(axes(A,1)),local_range)
  col_positions,new_col_partition = _restrict_to_local_range(partition(axes(A,2)),local_range)
  new_values = map(partition(A),row_positions,col_positions) do values,rp,cp
    values[rp,cp]
  end
  PSparseMatrix(new_values,new_row_partition,new_col_partition)
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