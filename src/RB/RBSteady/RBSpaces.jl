"""
    reduced_spaces(solver::RBSolver,feop::ParamOperator,s::AbstractSnapshots
      ) -> (RBSpace, RBSpace)

Computes the subspace of the test, trial `FESpace`s contained in the FE operator
`feop` by compressing the snapshots `s`
"""
function reduced_spaces(solver::RBSolver,feop::ParamOperator,s::AbstractSnapshots)
  red = get_state_reduction(solver)
  soff = select_snapshots(s,offline_params(solver))
  reduced_spaces(red,feop,soff)
end

function reduced_spaces(red::Reduction,feop::ParamOperator,s::AbstractSnapshots)
  t = @timed begin
    basis = reduced_basis(red,feop,s)
  end
  println(CostTracker(t,name="Basis construction"))

  reduced_trial = reduced_subspace(get_trial(feop),basis)
  reduced_test = reduced_subspace(get_test(feop),basis)
  return reduced_trial,reduced_test
end

"""
    reduced_basis(red::Reduction,s::AbstractSnapshots,args...) -> Projection

Computes the basis by compressing the snapshots `s`
"""
function reduced_basis(
  red::Reduction,
  s::AbstractSnapshots,
  args...)

  projection(red,s,args...)
end

function reduced_basis(
  red::Reduction,
  feop::ParamOperator,
  s::AbstractSnapshots)

  reduced_basis(red,s)
end

function reduced_basis(
  red::Reduction{<:ReductionStyle,EnergyNorm},
  feop::ParamOperator,
  s::AbstractSnapshots)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  reduced_basis(red,s,norm_matrix)
end

function reduced_basis(
  red::SupremizerReduction,
  feop::ParamOperator,
  s::AbstractSnapshots)

  norm_matrix = assemble_matrix(feop,get_norm(red))
  supr_matrix = assemble_matrix(feop,get_supr(red))
  basis = reduced_basis(get_reduction(red),s,norm_matrix)
  enrich!(red,basis,norm_matrix,supr_matrix)
  return basis
end

"""
    reduced_subspace(space::FESpace,basis::Projection) -> RBSpace

Generic constructor of a [`RBSpace`](@ref) from a `FESpace` `space` and a projection
`basis`
"""
function reduced_subspace(space::FESpace,basis::Projection)
  @abstractmethod
end

"""
    abstract type RBSpace <: FESpace end

Represents a vector subspace of a `FESpace`.

Subtypes:

- [`SingleFieldRBSpace`](@ref)
- [`MultiFieldRBSpace`](@ref)
"""
abstract type RBSpace{S} <: FESpace end

function Arrays.evaluate(r::RBSpace,args...)
  space = evaluate(get_fe_space(r),args...)
  reduced_subspace(space,get_reduced_subspace(r))
end

function Arrays.evaluate(r::RBSpace,::Nothing)
  space = evaluate(get_fe_space(r),nothing)
  reduced_subspace(space,get_reduced_subspace(r))
end

(U::RBSpace)(μ) = evaluate(U,μ)

FESpaces.get_fe_space(r::RBSpace) = @abstractmethod

"""
    get_reduced_subspace(r::RBSpace) -> Projection

Returns the [`Projection`](@ref) spanning the reduced subspace contained in `r`
"""
get_reduced_subspace(r::RBSpace) = @abstractmethod

get_basis(r::RBSpace) = get_basis(get_reduced_subspace(r))
num_fe_dofs(r::RBSpace) = num_free_dofs(get_fe_space(r))
num_reduced_dofs(r::RBSpace) = num_reduced_dofs(get_reduced_subspace(r))

FESpaces.get_triangulation(r::RBSpace) = get_triangulation(get_fe_space(r))
FESpaces.get_free_dof_ids(r::RBSpace) = get_free_dof_ids(get_fe_space(r))
FESpaces.get_dof_value_type(r::RBSpace) = get_dof_value_type(get_fe_space(r))
FESpaces.get_cell_dof_ids(r::RBSpace) = get_cell_dof_ids(get_fe_space(r))
FESpaces.get_fe_basis(r::RBSpace) = get_fe_basis(get_fe_space(r))
FESpaces.get_trial_fe_basis(r::RBSpace) = get_trial_fe_basis(get_fe_space(r))
FESpaces.get_fe_dof_basis(r::RBSpace) = get_fe_dof_basis(get_fe_space(r))
FESpaces.ConstraintStyle(r::RBSpace) = ConstraintStyle(get_fe_space(r))
FESpaces.get_cell_isconstrained(r::RBSpace) = get_cell_isconstrained(get_fe_space(r))
FESpaces.get_cell_constraints(r::RBSpace) = get_cell_constraints(get_fe_space(r))
FESpaces.get_vector_type(r::RBSpace) = typeof(zero_free_values(r))
FESpaces.get_dirichlet_dof_ids(r::RBSpace) = get_dirichlet_dof_ids(get_fe_space(r))
FESpaces.get_cell_is_dirichlet(r::RBSpace) = get_cell_is_dirichlet(get_fe_space(r))
FESpaces.num_dirichlet_tags(r::RBSpace) = num_dirichlet_tags(get_fe_space(r))
FESpaces.get_dirichlet_dof_tag(r::RBSpace) = get_dirichlet_dof_tag(get_fe_space(r))
DofMaps.get_dof_map(r::RBSpace) = get_dof_map(get_fe_space(r))
ParamDataStructures.param_length(r::RBSpace) = param_length(get_fe_space(r))

function FESpaces.zero_free_values(r::RBSpace)
  x = zero_free_values(get_fe_space(r))
  x̂ = project(r,x)
  reduced_vector(x̂,x)
end

FESpaces.zero_dirichlet_values(r::RBSpace) = zero_dirichlet_values(get_fe_space(r))

function FESpaces.scatter_free_and_dirichlet_values(r::RBSpace,fv,dv)
  scatter_free_and_dirichlet_values(get_fe_space(r),fv.fe_data,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,r::RBSpace,cv)
  gather_free_and_dirichlet_values!(fv.fe_data,dv,get_fe_space(r),cv)
end

for (f,f!) in zip((:project,:inv_project),(:project!,:inv_project!))
  @eval begin
    function $f(r::RBSpace,x::AbstractVector)
      $f(get_reduced_subspace(r),x)
    end

    function $f!(y,r::RBSpace,x::AbstractVector)
      $f!(y,get_reduced_subspace(r),x)
    end
  end
end

function project(r::RBSpace,a::AbstractRBVector)
  project!(a.data,r,a.fe_data)
  return a.data
end

function inv_project(r::RBSpace,a::AbstractRBVector)
  inv_project!(a.fe_data,r,a.data)
  return a.fe_data
end

function project(r::RBSpace,x::Projection)
  galerkin_projection(get_reduced_subspace(r),x)
end

function project(r1::RBSpace,x::Projection,r2::RBSpace)
  galerkin_projection(get_reduced_subspace(r1),x,get_reduced_subspace(r2))
end

function FESpaces.FEFunction(r::RBSpace,x̂::AbstractVector)
  x = inv_project(r,x̂)
  fe = get_fe_space(r)
  xdir = get_dirichlet_dof_values(fe)
  return FEFunction(fe,x,xdir)
end

"""
    struct SingleFieldRBSpace{S<:SingleFieldFESpace} <: RBSpace{S}
      space::S
      subspace::Projection
    end

Reduced basis subspace of a `SingleFieldFESpace` in [`Gridap`](@ref)
"""
struct SingleFieldRBSpace{S<:SingleFieldFESpace} <: RBSpace{S}
  space::S
  subspace::Projection
end

function reduced_subspace(space::SingleFieldFESpace,basis::Projection)
  SingleFieldRBSpace(space,basis)
end

FESpaces.get_fe_space(r::SingleFieldRBSpace) = r.space
get_reduced_subspace(r::SingleFieldRBSpace) = r.subspace

"""
    struct MultiFieldRBSpace{S<:MultiFieldFESpace} <: RBSpace{S}
      space::S
      subspace::BlockProjection
    end

Reduced basis subspace of a `MultiFieldFESpace` in [`Gridap`](@ref)
"""
struct MultiFieldRBSpace{S<:MultiFieldFESpace} <: RBSpace{S}
  space::S
  subspace::BlockProjection
end

function reduced_subspace(space::MultiFieldFESpace,subspace::BlockProjection)
  MultiFieldRBSpace(space,subspace)
end

FESpaces.get_fe_space(r::MultiFieldRBSpace) = r.space
get_reduced_subspace(r::MultiFieldRBSpace) = r.subspace

function Base.getindex(r::MultiFieldRBSpace,i::Integer)
  mfe = get_fe_space(r)
  rsp = get_reduced_subspace(r)
  return reduced_subspace(mfe.spaces[i],rsp[i])
end

function Base.iterate(r::MultiFieldRBSpace,state=1)
  if state > num_fields(r)
    return nothing
  end
  mfe = get_fe_space(r)
  rsp = get_reduced_subspace(r)
  ri = reduced_subspace(mfe[state],rsp[state])
  return ri,state+1
end

MultiField.MultiFieldStyle(r::MultiFieldRBSpace) = MultiFieldStyle(get_fe_space(r))
MultiField.num_fields(r::MultiFieldRBSpace) = num_fields(get_fe_space(r))
Base.length(r::MultiFieldRBSpace) = num_fields(r)

function FESpaces.zero_free_values(r::MultiFieldRBSpace)
  x̂ = mortar(map(zero_free_values,r))
  unfold(x̂)
end

# utils

function to_snapshots(f::RBSpace,x̂::AbstractParamVector,r::AbstractRealization)
  fr = f(r)
  x = inv_project(fr,x̂)
  i = get_dof_map(fr)
  Snapshots(x,i,r)
end

function projection_error(f::RBSpace,x::AbstractParamVector,r::AbstractRealization)
  fr = f(r)
  i = get_dof_map(fr)
  a = get_reduced_subspace(f)
  x̂ = project(fr,x)
  x′ = inv_project(fr,x̂)
  s = Snapshots(x,i,r)
  s′ = Snapshots(x′,i,r)
  return compute_relative_error(s,s′,get_norm_matrix(a))
end
