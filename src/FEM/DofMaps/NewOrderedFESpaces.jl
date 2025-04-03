"""
    abstract type OrderedFESpace{S} <: SingleFieldFESpace end

Interface for FE spaces that feature a DOF reordering
"""
abstract type OrderedFESpace{S} <: SingleFieldFESpace end

# FESpace interface

FESpaces.get_fe_space(f::OrderedFESpace) = @abstractmethod

"""
    get_cell_odof_ids(f::OrderedFESpace) -> AbstractArray

Fetches the ordered connectivity structure
"""
get_cell_odof_ids(f::OrderedFESpace) = @abstractmethod

FESpaces.ConstraintStyle(::Type{<:OrderedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::OrderedFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::OrderedFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::OrderedFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::OrderedFESpace) = get_cell_odof_ids(f)

FESpaces.get_fe_basis(f::OrderedFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::OrderedFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::OrderedFESpace) = get_fe_dof_basis(get_fe_space(f))

FESpaces.get_cell_isconstrained(f::OrderedFESpace) = get_cell_isconstrained(get_fe_space(f))

FESpaces.get_cell_constraints(f::OrderedFESpace) = get_cell_constraints(get_fe_space(f))

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_is_dirichlet(f::OrderedFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::OrderedFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::OrderedFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::OrderedFESpace) = get_dirichlet_dof_tag(get_fe_space(f))

FESpaces.get_vector_type(f::OrderedFESpace) = get_vector_type(get_fe_space(f))

# Scatters correctly ordered free and dirichlet values
function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpace,fv,dv)
  cell_odof_ids = get_cell_odof_ids(f)
  cell_ovalues = lazy_map(Broadcasting(PosNegReindex(fv,dv)),cell_odof_ids)
  cell_ovalue_to_value(f,cell_ovalues)
end

# Gathers correctly ordered free and dirichlet values
function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::OrderedFESpace,cv)
  cell_ovals = cell_value_to_ovalue(f,cv)
  cell_odofs = get_cell_odof_ids(f)
  cache_ovals = array_cache(cell_ovals)
  cache_odofs = array_cache(cell_odofs)
  cells = 1:length(cell_ovals)

  FESpaces._free_and_dirichlet_values_fill!(
    fv,
    dv,
    cache_ovals,
    cache_odofs,
    cell_ovals,
    cell_odofs,
    cells)

  (fv,dv)
end

function CartesianFESpace(f::SingleFieldFESpace)
  trian = get_triangulation(f)
  model = get_active_model(trian)
  CartesianFESpace(model,f)
end

function CartesianFESpace(f_bg::SingleFieldFESpace,f_act::FESpaceWithLinearConstraints)
  trian_bg = get_triangulation(f_bg)
  model_bg = get_active_model(trian_bg)
  CartesianFESpace(model_bg,f_bg,f_act)
end

function CartesianFESpace(::DiscreteModel,::SingleFieldFESpace...)
  @notimplemented "Background model must be cartesian the selected dof reordering"
end

function CartesianFESpace(::CartesianDiscreteModel,f::SingleFieldFESpace)
  cell_odofs_ids = get_cell_odof_ids(f)
  bg_odofs_to_odofs = get_bg_odof_to_odof(f,cell_odofs_ids)
  odofs_to_bg_odofs = findall(!iszero,bg_odofs_to_odofs)
  CartesianFESpace(f,cell_odofs_ids,bg_odofs_to_odofs,odofs_to_bg_odofs)
end

# I need 4 dof maps:
# 1) from aggregate ordered dofs, to background ordered dofs
# 2) from background ordered dofs, to aggregate ordered dofs
# 3) from external aggregate ordered dofs, to background ordered dofs
# 4) from external aggregate ordered dofs, to external active ordered dofs

function CartesianFESpace(
  ::CartesianDiscreteModel,
  f_bg::SingleFieldFESpace,
  f_agg::FESpaceWithLinearConstraints,
  f_act_out::SingleFieldFESpace
  )

  cell_odofs_ids = get_cell_odof_ids(f_bg)
  bg_dof_to_bg_odof = reorder_dofs(f_bg,cell_odofs_ids)

  # # map 1
  # agg_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_agg)
  # agg_dof_to_bg_odof = reorder_dof_map(bg_dof_to_bg_odof,agg_dof_to_bg_dof)

  # map 2
  agg_bg_dof_to_dof = get_bg_dof_to_dof(f_bg,f_agg)
  agg_bg_dof_to_odof = reorder_dof_map(bg_dof_to_bg_odof,agg_bg_dof_to_dof)

  # map 1
  agg_dof_to_bg_odof = findall(!iszero,agg_bg_dof_to_odof)

  # map 3
  act_out_dof_to_bg_dof = get_dof_to_bg_dof(f_bg,f_act_out)
  act_out_dof_to_bg_odof = reorder_dof_map(bg_dof_to_bg_odof,act_out_dof_to_bg_dof)
  agg_out_dof_to_bg_odof = setdiff(act_out_dof_to_bg_odof,agg_dof_to_bg_odof)

  # map 4
  for (iagg,bg_odof) in enumerate(agg_out_dof_to_bg_odof)
    act_odof = findfirst(act_out_dof_to_bg_odof.==bg_odof)
    agg_out_dof_to_act_out_odof[iagg] = act_odof
  end

  # aggregated cell odof ids
  cell_odof_ids =

  ExtensionCartesianFESpace(
    f_agg,
    agg_dof_to_bg_odof,
    agg_bg_dof_to_odof,
    agg_out_dof_to_bg_odof,
    agg_out_dof_to_act_out_odof)
end
