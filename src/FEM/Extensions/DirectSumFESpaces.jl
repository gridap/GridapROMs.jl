struct DirectSumFESpace{S<:EmbeddedFESpace}
  space::S
  complementary::EmbeddedFESpace
  function DirectSumFESpace(space::S,complementary::EmbeddedFESpace) where S<:EmbeddedFESpace
    _check_intersection(space,complementary)
    new{S}(space,complementary)
  end
end

function DirectSumFESpace(
  bg_space::SingleFieldFESpace,
  act_space::SingleFieldFESpace,
  inact_space::SingleFieldFESpace
  )

  space = EmbeddedFESpace(bg_space,act_space)
  complementary = get_complementary(space,inact_space)
  DirectSumFESpace(space,complementary)
end

function DirectSumFESpace(
  bg_space::SingleFieldFESpace,
  act_space::SingleFieldFESpace
  )

  @notimplemented "Write a setdiff for FE spaces"
end

FESpaces.ConstraintStyle(::Type{<:DirectSumFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::DirectSumFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::DirectSumFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::DirectSumFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::DirectSumFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::DirectSumFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::DirectSumFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::DirectSumFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::DirectSumFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::DirectSumFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::DirectSumFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::DirectSumFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::DirectSumFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::DirectSumFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::DirectSumFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::DirectSumFESpace) = get_vector_type(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::DirectSumFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::DirectSumFESpace,cv)
  FESpaces.gather_free_and_dirichlet_values(f.space,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::DirectSumFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end

# utils

get_emb_space(f::DirectSumFESpace) = get_emb_space(f.space)
get_act_space(f::DirectSumFESpace) = get_act_space(f.space)
get_bg_space(f::DirectSumFESpace) = get_bg_space(f.space)

function get_cells_to_bg_cells(f::SingleFieldFESpace)
  get_cell_to_bg_cell(f.space)
end

function get_bg_cells_to_cells(f::SingleFieldFESpace)
  get_bg_cell_to_cell(f.space)
end

function get_complem_cells_to_bg_cells(f::SingleFieldFESpace)
  get_cell_to_bg_cell(f.complementary)
end

function get_bg_cells_to_complem_cells(f::SingleFieldFESpace)
  get_bg_cell_to_cell(f.complementary)
end

function (⊕)(uh::FEFunction,vh::FEFunction)
  space = get_fe_space(uh)
  complementary = get_fe_space(vh)
  @check _check_intersection(space,complementary)

  bg_fv = zero_bg_free_values(space)
  bg_dv = zero_bg_dirichlet_values(space)

  fv = get_free_dof_values(uh)
  dv = get_dirichlet_dof_values(uh)
  _bg_vals_from_vals!(bg_fv,bg_dv,space,fv,dv)

  cfv = get_free_dof_values(vh)
  cdv = get_dirichlet_dof_values(vh)
  _bg_vals_from_vals!(bg_fv,bg_dv,complementary,cfv,cdv)

  FEFunction(get_bg_space(uh),bg_fv,bg_dv)
end
