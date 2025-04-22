struct DirectSumFESpace{S<:EmbeddedFESpace} <: SingleFieldFESpace
  space::S
  complementary::EmbeddedFESpace
end

function DirectSumFESpace(bg_space::SingleFieldFESpace,act_space::SingleFieldFESpace)
  space = EmbeddedFESpace(act_space,bg_space)
  complementary = complementary_space(space)
  DirectSumFESpace(space,complementary)
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
get_fdof_to_bg_fdof(f::DirectSumFESpace) = get_fdof_to_bg_fdof(f.space)
get_ddof_to_bg_ddof(f::DirectSumFESpace) = get_ddof_to_bg_ddof(f.space)
get_active_fdof_to_bg_fdof(f::DirectSumFESpace) = get_active_fdof_to_bg_fdof(f.space)
get_active_ddof_to_bg_ddof(f::DirectSumFESpace) = get_active_ddof_to_bg_ddof(f.space)

get_space(f::DirectSumFESpace) = f.space
get_out_space(f::DirectSumFESpace) = f.complementary

for f in (:get_space,:get_out_space)
  for T in (:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
    @eval begin
      $f(f::$T) = $f(f.space)
    end
  end

  @eval begin
    function $f(f::TrivialParamFESpace)
      TrivialParamFESpace($f(f.space),f.plength)
    end

    function $f(f::TrialParamFESpace)
      TrialParamFESpace($f(f.space),f.dirichlet_values)
    end
  end
end

function get_cells_to_bg_cells(f::DirectSumFESpace)
  get_cell_to_bg_cell(f.space)
end

function get_bg_cells_to_cells(f::DirectSumFESpace)
  get_bg_cell_to_cell(f.space)
end

function get_complem_cells_to_bg_cells(f::DirectSumFESpace)
  get_cell_to_bg_cell(f.complementary)
end

function get_bg_cells_to_complem_cells(f::DirectSumFESpace)
  get_bg_cell_to_cell(f.complementary)
end

function (⊕)(uh::FEFunction,vh::FEFunction)
  space = get_fe_space(uh)
  complementary = get_fe_space(vh)
  @check _same_background_space(space,complementary)

  bg_fv = zero_bg_free_values(space)
  bg_dv = zero_bg_dirichlet_values(space)

  cfv = get_free_dof_values(vh)
  cdv = get_dirichlet_dof_values(vh)
  _bg_vals_from_vals!(bg_fv,bg_dv,complementary,cfv,cdv)

  fv = get_free_dof_values(uh)
  dv = get_dirichlet_dof_values(uh)
  _bg_vals_from_vals!(bg_fv,bg_dv,space,fv,dv)

  FEFunction(get_bg_space(space),bg_fv,bg_dv)
end

function _same_background_space(space::SingleFieldFESpace,complementary::SingleFieldFESpace)
  get_bg_space(space)==get_bg_space(complementary)
end

for T in (:SingleFieldParamFESpace,:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
  @eval begin
    function _same_background_space(space::$T,complementary::$T)
      _same_background_space(space.space,complementary.space)
    end
  end
end

FESpaces.get_dirichlet_dof_values(uh::SingleFieldFEFunction) = uh.dirichlet_values
FESpaces.get_dirichlet_dof_values(uh::SingleFieldParamFEFunction) = uh.dirichlet_values

# dof map utils

function DofMaps.get_dof_map(f::DirectSumFESpace,args...)
  get_dof_map(get_bg_space(f),args...)
end

function ParamSteady._assemble_matrix(f,U::FESpace,V::DirectSumFESpace)
  ParamSteady._assemble_matrix(f,get_bg_space(U),get_bg_space(V))
end

function DofMaps.SparsityPattern(
  U::SingleFieldFESpace,
  V::DirectSumFESpace,
  trian=DofMaps._get_common_domain(U,V)
  )

  a = ExtensionAssembler(U,V)
  m1 = nz_counter(FESpaces.get_matrix_builder(a),(FESpaces.get_rows(a),FESpaces.get_cols(a)))
  cellidsrows = get_bg_cell_dof_ids(V,trian)
  cellidscols = get_bg_cell_dof_ids(U,trian)
  DofMaps.trivial_symbolic_loop_matrix!(m1,cellidsrows,cellidscols)
  m2 = Algebra.nz_allocation(m1)
  DofMaps.trivial_symbolic_loop_matrix!(m2,cellidsrows,cellidscols)
  m3 = Algebra.create_from_nz(m2)
  SparsityPattern(m3)
end
