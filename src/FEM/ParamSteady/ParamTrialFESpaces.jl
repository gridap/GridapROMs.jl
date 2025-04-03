"""
    struct UnEvalTrialFESpace{A,B} <: SingleFieldFESpace
      space::A
      space0::B
      dirichlet::Union{Function,AbstractVector{<:Function}}
    end

Struct representing trial FE spaces that are not evaluated yet. This may include
FE spaces representing transient problems (although the implementation in `Gridap`,
called `TransientTrialFESpace`, does not fall into this category), parametric
problems, and transient-parametric problems.
"""
struct UnEvalTrialFESpace{A,B} <: SingleFieldFESpace
  space::A
  space0::B
  dirichlet::Union{Function,AbstractVector{<:Function}}

  function UnEvalTrialFESpace(
    space::A,
    dirichlet::Union{Function,AbstractVector{<:Function}}) where A

    space0 = HomogeneousTrialFESpace(space)
    B = typeof(space0)
    new{A,B}(space,space0,dirichlet)
  end
end

"""
    const ParamTrialFESpace = UnEvalTrialFESpace
"""
const ParamTrialFESpace = UnEvalTrialFESpace

function ParamTrialFESpace(space)
  dof = get_fe_dof_basis(space)
  T = get_dof_eltype(dof)
  function _param_zero(μ::Realization)
    z(μ) = x -> zero(T)
    ParamFunction(z,μ)
  end
  function _param_zero(μ::Realization,t)
    z(μ,t) = x -> zero(T)
    TransientParamFunction(z,μ,t)
  end
  UnEvalTrialFESpace(space,_param_zero)
end

# FE space interface

FESpaces.ConstraintStyle(::Type{<:UnEvalTrialFESpace{A}}) where A = ConstraintStyle(A)
FESpaces.get_free_dof_ids(f::UnEvalTrialFESpace) = get_free_dof_ids(f.space)
FESpaces.get_vector_type(f::UnEvalTrialFESpace) = get_vector_type(f.space)
CellData.get_triangulation(f::UnEvalTrialFESpace) = get_triangulation(f.space)
FESpaces.get_cell_dof_ids(f::UnEvalTrialFESpace) = get_cell_dof_ids(f.space)
FESpaces.get_fe_basis(f::UnEvalTrialFESpace) = get_fe_basis(f.space)
FESpaces.get_fe_dof_basis(f::UnEvalTrialFESpace) = get_fe_dof_basis(f.space)
FESpaces.get_cell_constraints(f::UnEvalTrialFESpace) = get_cell_constraints(f.space)
FESpaces.get_cell_isconstrained(f::UnEvalTrialFESpace) = get_cell_isconstrained(f.space)
FESpaces.get_dirichlet_dof_ids(f::UnEvalTrialFESpace) = get_dirichlet_dof_ids(f.space)
FESpaces.num_dirichlet_tags(f::UnEvalTrialFESpace) = num_dirichlet_tags(f.space)
FESpaces.get_dirichlet_dof_tag(f::UnEvalTrialFESpace) = get_dirichlet_dof_tag(f.space)
function FESpaces.scatter_free_and_dirichlet_values(f::UnEvalTrialFESpace,free_values,dirichlet_values)
  scatter_free_and_dirichlet_values(f.space,free_values,dirichlet_values)
end
function FESpaces.gather_free_and_dirichlet_values!(free_values,dirichlet_values,f::UnEvalTrialFESpace,cell_vals)
  gather_free_and_dirichlet_values!(free_values,dirichlet_values,f.space,cell_vals)
end

function FESpaces.get_dirichlet_dof_values(f::UnEvalTrialFESpace)
  msg = """
  It does not make sense to get the Dirichlet DOF values of a transient FE space. You
  should first evaluate the transient FE space at a point in time and get the Dirichlet
  DOF values from there.
  """
  @unreachable msg
end

for F in (:TrialFESpace,:TransientTrialFESpace,:UnEvalTrialFESpace)
  @eval begin
    function DofMaps.get_dof_map(trial::$F,args...)
      get_dof_map(trial.space,args...)
    end

    function DofMaps.get_sparse_dof_map(trial::$F,test::SingleFieldFESpace,args...)
      get_sparse_dof_map(trial.space,test,args...)
    end
  end
end

# Evaluations

function ODEs.allocate_space(U::UnEvalTrialFESpace,r::Realization)
  HomogeneousTrialParamFESpace(U.space,length(r))
end

function Arrays.evaluate(U::UnEvalTrialFESpace,args...)
  Upt = allocate_space(U,args...)
  evaluate!(Upt,U,args...)
  Upt
end

function Arrays.evaluate!(Upt::TrialParamFESpace,U::UnEvalTrialFESpace,r::Realization)
  dir(f) = f(r)
  dir(f::Vector) = dir.(f)
  TrialParamFESpace!(Upt,dir(U.dirichlet))
  Upt
end

(U::UnEvalTrialFESpace)(r) = evaluate(U,r)
Arrays.evaluate(U::UnEvalTrialFESpace,r::Nothing) = U.space0

# Define the UnEvalTrialFESpace interface for stationary spaces

ODEs.allocate_space(U::FESpace,r) = U
Arrays.evaluate!(Upt::FESpace,U::FESpace,r::AbstractRealization) = U
Arrays.evaluate(U::FESpace,r) = U

# Define the interface for MultiField

function has_param(U::MultiFieldFESpace)
  any(space -> space isa UnEvalTrialFESpace,U.spaces)
end

function ODEs.allocate_space(U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  spaces = map(U->allocate_space(U,r),U.spaces)
  style = MultiFieldStyle(U)
  MultiFieldParamFESpace(spaces;style)
end

function Arrays.evaluate!(Upt::MultiFieldFESpace,U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  for (Upti,Ui) in zip(Upt,U)
    evaluate!(Upti,Ui,r)
  end
  Upt
end

function Arrays.evaluate(U::MultiFieldFESpace,r::Realization)
  if !has_param(U)
    return U
  end
  Ut = allocate_space(U,r)
  evaluate!(Ut,U,r)
  Ut
end
