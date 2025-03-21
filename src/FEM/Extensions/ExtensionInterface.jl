abstract type ExtensionStyle end
struct ZeroExtension <: ExtensionStyle end
struct FunctionExtension <: ExtensionStyle end
struct HarmonicExtension <: ExtensionStyle end

abstract type Extension{E<:ExtensionStyle} end

struct GenericExtension{E<:ExtensionStyle} <: Extension{E}
  style::E
  matrix::AbstractMatrix
  vector::AbstractVector
  values::FEFunction
  dof_to_bg_dofs::AbstractVector
end

function Extension(
  style::FunctionExtension,
  space::SingleFieldFESpace,
  dof_to_bg_dofs::AbstractVector
  )

  matrix = _build_mass_matrix(space)
  vector = zero_free_values(space)
  values = zero_free_values(space)
  zh = FEFunction(space,values)
  GenericExtension(style,matrix,vector,zh,dof_to_bg_dofs)
end

function Extension(
  style::FunctionExtension,
  space::SingleFieldFESpace,
  dof_to_bg_dofs::AbstractVector,
  f::Function
  )

  matrix = _build_mass_matrix(space)
  fh = interpolate_everywhere(f,space)
  vector = get_free_dof_values(fh)
  GenericExtension(style,matrix,vector,fh,dof_to_bg_dofs)
end

function Extension(
  style::HarmonicExtension,
  space::SingleFieldFESpace,
  dof_to_bg_dofs::AbstractVector,
  a::Function,
  l::Function,
  )

  laplacian = assemble_matrix(a,space,space)
  vector = assemble_vector(l,space)
  factor = lu(laplacian)
  values = similar(vector)
  ldiv!(values,factor,vector)
  vh = FEFunction(space,values)
  GenericExtension(style,laplacian,vector,vh,dof_to_bg_dofs)
end

function Extension(style::ExtensionStyle,bg_space::SingleFieldFESpace,space::SingleFieldFESpace,args...)
  dof_to_bg_dofs = get_dof_to_bg_dof(bg_space,space)
  Extension(style,space,dof_to_bg_dofs,args...)
end

function Extension(style::ExtensionStyle,args...)
  @abstractmethod
end

FESpaces.get_fe_space(ext::GenericExtension) = ext.values.fe_space

function FESpaces.scatter_free_and_dirichlet_values(ext::GenericExtension)
  scatter_free_and_dirichlet_values(
    get_fe_space(ext),ext.values.free_values,ext.values.dirichlet_values)
end

function FESpaces.gather_free_and_dirichlet_values(ext::GenericExtension)
  (ext.values.free_values,ext.values.dirichlet_values)
end

get_out_dof_to_bg_dofs(ext::GenericExtension) = ext.dof_to_bg_dofs

function _build_mass_matrix(space::SingleFieldFESpace)
  Ω = get_triangulation(space)
  degree = 2*get_polynomial_order(space)
  dΩ = Measure(Ω,degree)
  mass(u,v) = ∫(u⋅v)dΩ
  assemble_matrix(mass,space,space)
end
