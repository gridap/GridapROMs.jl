struct ConfigurationFESpace{A<:SingleFieldFESpace,B<:ConfigurationStyle} <: SingleFieldFESpace
  space::A
  style::B
end

function get_configuration_space(style::ConfigurationStyle,f::SingleFieldFESpace)
  ConfigurationFESpace(f,style)
end

function get_configuration_space(style::ConfigurationStyle,f::UnEvalTrialFESpace)
  UnEvalTrialFESpace(get_configuration_space(style,f.space),f.dirichlet)
end

function get_configuration_space(style::ConfigurationStyle,f::TrialParamFESpace)
  TrialParamFESpace(f.dirichlet_values,get_configuration_space(style,f.space))
end

function get_configuration_space(style::ConfigurationStyle,f::TrivialParamFESpace)
  TrivialParamFESpace(get_configuration_space(style,f.space),f.plength)
end

function get_configuration_space(style::ConfigurationStyle,f::MultiFieldFESpace)
  spaces′ = map(s -> get_configuration_space(style,s),f.spaces)
  MultiFieldFESpace(f.vector_type,spaces′,f.multi_field_style)
end

function get_configuration_space(f::FESpace)
  get_configuration_space(ReferenceConfiguration(),f)
end

function get_configuration_space(f::FESpace,index::Int)
  get_configuration_space(ConfigurationAtIndex(index),f)
end

function FESpaces.get_triangulation(f::ConfigurationFESpace)
  get_configuration_triangulation(f.style,get_triangulation(f.space))
end

for F in (:(FESpaces.get_fe_basis),:(FESpaces.get_fe_dof_basis))
  @eval begin
    function $F(f::ConfigurationFESpace)
      a = $F(f.space)
      trian = get_triangulation(f)
      change_domain(a,trian,DomainStyle(a))
    end
  end
end

FESpaces.get_free_dof_ids(f::ConfigurationFESpace) = get_free_dof_ids(f.space)

FESpaces.zero_free_values(f::ConfigurationFESpace) = zero_free_values(f.space)

FESpaces.get_dof_value_type(f::ConfigurationFESpace) = get_dof_value_type(f.space)

FESpaces.get_vector_type(f::ConfigurationFESpace) = get_vector_type(f.space)

FESpaces.get_cell_dof_ids(f::ConfigurationFESpace) = get_cell_dof_ids(f.space)

FESpaces.ConstraintStyle(::Type{<:ConfigurationFESpace{A}}) where A = ConstraintStyle(A)

FESpaces.get_cell_isconstrained(f::ConfigurationFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::ConfigurationFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::ConfigurationFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::ConfigurationFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.zero_dirichlet_values(f::ConfigurationFESpace) = zero_dirichlet_values(f.space)

FESpaces.num_dirichlet_tags(f::ConfigurationFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::ConfigurationFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::ConfigurationFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

FESpaces.gather_free_and_dirichlet_values(f::ConfigurationFESpace,cv) = gather_free_and_dirichlet_values(f.space,cv)

FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::ConfigurationFESpace,cv) = gather_free_and_dirichlet_values!(fv,dv,f.space,cv)

FESpaces.gather_dirichlet_values(f::ConfigurationFESpace,cv) = gather_dirichlet_values(f.space,cv)

FESpaces.gather_dirichlet_values!(dv,f::ConfigurationFESpace,cv) = gather_dirichlet_values!(dv,f.space,cv)

FESpaces.gather_free_values(f::ConfigurationFESpace,cv) = gather_free_values(f.space,cv)

FESpaces.gather_free_values!(fv,f::ConfigurationFESpace,cv) = gather_free_values!(fv,f.space,cv)
