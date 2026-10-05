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
# operator on field_fe_space[i] directly), so it isn't generically
# splittable into per-field forms here: it is assembled once on U,V exactly
# as given, then each field's diagonal block is cut out via the same
# per-rank-LOCAL-range restriction used for state snapshots (Snapshots.jl's
# _local_ranges_from_sizes/_restrict_rows_to_local_range) -- the sizes here
# come from each field's own get_free_dof_ids instead of a dof map, but it's
# the same per-rank local (own+ghost) dof count either way, so the two
# agree. This also matches field_fe_space[i]'s own standalone numbering:
# both are the result of the same deterministic rank-ordered sequential
# assignment to a field's own entries (which is, in fact, exactly how
# GridapDistributed's generate_multi_field_gids derives a BlockMultiFieldStyle
# block's gids for a single-field block -- it just reuses
# get_free_dof_ids(field_fe_space[i]) directly), so this is consistent with
# the BlockOperator path below too. A field-major GLOBAL offset range (the
# previous approach here, and what the serial-only
# ParamDataStructures.offset_indices computes) is wrong in general: see the
# note in Snapshots.jl's _local_field_ranges for why.
function _assemble_operator(op::EnergyNorm,U::DistributedMultiFieldFESpace,V::DistributedMultiFieldFESpace)
  A = assemble_matrix(op.form,U,V)
  sizes = map(Ui -> map(local_length,partition(get_free_dof_ids(Ui))),U.field_fe_space)
  local_ranges = _local_ranges_from_sizes(sizes)
  map(lr -> _restrict_diag_block(A,lr),local_ranges)
end

function _restrict_diag_block(A::PSparseMatrix,local_range)
  row_positions,new_row_partition = _restrict_rows_to_local_range(partition(axes(A,1)),local_range)
  col_positions,new_col_partition = _restrict_rows_to_local_range(partition(axes(A,2)),local_range)
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