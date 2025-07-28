struct BaseConfigurationFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
end

function get_configuration_space(f::SingleFieldFESpace)
  BaseConfigurationFESpace(f)
end

function get_configuration_space(f::MultiFieldFESpace)
  spaces′ = map(get_configuration_space,f.spaces)
  MultiFieldFESpace(f.vector_type,spaces′,f.multi_field_style)
end

FESpaces.get_triangulation(f::BaseConfigurationFESpace) = get_base_triangulation(f.space)

get_base_triangulation(f::SingleFieldFESpace) = get_base_triangulation(get_triangulation(f))

for F in (:(FESpaces.get_fe_basis),:(FESpaces.get_trial_fe_basis),:(FESpaces.get_fe_dof_basis))
  @eval begin
    function $F(f::BaseConfigurationFESpace)
      a = $F(f.space)
      trian = get_triangulation(f)
      change_domain(a,trian,DomainStyle(a))
    end
  end
end

FESpaces.get_free_dof_ids(f::BaseConfigurationFESpace) = get_free_dof_ids(f.space)

FESpaces.zero_free_values(f::BaseConfigurationFESpace) = zero_free_values(f.space)

FESpaces.get_dof_value_type(f::BaseConfigurationFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::BaseConfigurationFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::BaseConfigurationFESpace) = get_cell_dof_ids(f.space)

FESpaces.ConstraintStyle(::Type{<:BaseConfigurationFESpace{B}}) where B = ConstraintStyle(B)

FESpaces.get_cell_isconstrained(f::BaseConfigurationFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::BaseConfigurationFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::BaseConfigurationFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::BaseConfigurationFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.zero_dirichlet_values(f::BaseConfigurationFESpace) = zero_dirichlet_values(f.space)

FESpaces.num_dirichlet_tags(f::BaseConfigurationFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::BaseConfigurationFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::BaseConfigurationFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(f::BaseConfigurationFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::BaseConfigurationFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

FESpaces.gather_dirichlet_values(f::BaseConfigurationFESpace,cv) = gather_dirichlet_values(f.space,cv)

FESpaces.gather_dirichlet_values!(dv,f::BaseConfigurationFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

FESpaces.gather_free_values(f::BaseConfigurationFESpace,cv) = gather_free_values(f.space,cv)

FESpaces.gather_free_values!(fv,f::BaseConfigurationFESpace,cv) = gather_free_values!(fv,f.space,cv)

struct BaseTriangulation{Dt,Dp} <: Triangulation{Dt,Dp}
  trian::Triangulation{Dt,Dp}
end

function get_base_triangulation(trian::Triangulation)
  BaseTriangulation(trian)
end

Geometry.get_background_model(t::BaseTriangulation) = get_background_model(t.trian)
Geometry.get_glue(t::BaseTriangulation{Dt},::Val{Dt}) where Dt = get_glue(t.trian,Val{Dt}())
Geometry.get_grid(t::BaseTriangulation) = _get_grid(get_grid(t.trian),get_background_model(t))

_get_grid(g::Grid,m::DiscreteModel) = g
_get_grid(g::ParamGrid,m::DiscreteModel) = @abstractmethod
_get_grid(g::ParamMappedGrid,m::DiscreteModel) = g.grid

function _get_grid(g::Geometry.GridPortion,m::DiscreteModel)
  Geometry.GridPortion(_get_grid(g.parent,m),g.cell_to_parent_cell)
end

function _get_grid(g::ParamUnstructuredGrid,m::DiscreteModel)
  node_coordinates = get_node_coordinates(m)
  UnstructuredGrid(
    node_coordinates,
    g.cell_node_ids,
    g.reffes,
    g.cell_type,
    g.orientation_style,
    g.facet_normal
  )
end
