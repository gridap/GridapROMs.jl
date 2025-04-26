"""
    struct TProductFESpace{S} <: SingleFieldFESpace
      space::S
      spaces_1d::Vector{<:SingleFieldFESpace}
      trian::TProductTriangulation
    end

Tensor product single field `FESpace`, storing a vector of 1-D `FESpace`s `spaces_1d`
of length D, and the D-dimensional `FESpace` `space` defined as their tensor product.
The tensor product triangulation `trian` is provided as a field to avoid
incompatibility issues when passing to `MultiField` scenarios
"""
struct TProductFESpace{S} <: SingleFieldFESpace
  space::S
  spaces_1d::Vector{<:SingleFieldFESpace}
  trian::TProductTriangulation
end

function FESpaces.FESpace(trian::TProductTriangulation,reffe::Tuple{<:ReferenceFEName,Any,Any};kwargs...)
  TProductFESpace(trian,reffe;kwargs...)
end

function TProductFESpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_background_model(trian)
  space = OrderedFESpace(model.model,reffe;kwargs...)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)

  TProductFESpace(space,spaces_1d,trian)
end

function TProductFESpace(
  space::FESpace,
  tptrian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args

  model = get_background_model(tptrian)
  cell_reffes_1d = map(model->ReferenceFE(model,basis,eltype(T),order;reffe_kwargs...),model.models_1d)
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)

  TProductFESpace(space,spaces_1d,tptrian)
end

function univariate_spaces(
  model::TProductDiscreteModel,
  cell_reffes;
  dirichlet_tags=Int[],
  dirichlet_masks=nothing,
  conformity=nothing,
  vector_type=nothing,
  kwargs...)

  if !isnothing(dirichlet_masks)
    for mask in dirichlet_masks
      !(all(mask) || !any(mask)) && _throw_tp_error()
    end
  end

  diri_tags_1d = get_1d_tags(model,dirichlet_tags)
  map(model.models_1d,cell_reffes,diri_tags_1d) do model,cell_reffe,tags
    OrderedFESpace(model,cell_reffe;dirichlet_tags=tags,conformity,vector_type)
  end
end

FESpaces.get_triangulation(f::TProductFESpace) = get_triangulation(f.space)

FESpaces.get_free_dof_ids(f::TProductFESpace) = get_free_dof_ids(f.space)

FESpaces.get_vector_type(f::TProductFESpace) = get_vector_type(f.space)

FESpaces.get_dof_value_type(f::TProductFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::TProductFESpace) = get_cell_dof_ids(f.space)

FESpaces.ConstraintStyle(::Type{<:TProductFESpace{A}}) where A = ConstraintStyle(A)

FESpaces.get_fe_basis(f::TProductFESpace) = get_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::TProductFESpace) = get_fe_dof_basis(f.space)

FESpaces.num_dirichlet_dofs(f::TProductFESpace) = num_dirichlet_dofs(f.space)

FESpaces.get_cell_isconstrained(f::TProductFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::TProductFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::TProductFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::TProductFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_tags(f::TProductFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::TProductFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.scatter_free_and_dirichlet_values(f::TProductFESpace,fv,dv) = scatter_free_and_dirichlet_values(f.space,fv,dv)

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::TProductFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end

function DofMaps.get_sparsity(U::TProductFESpace,V::TProductFESpace,args...)
  @check length(U.spaces_1d) == length(V.spaces_1d)
  sparsity = get_sparsity(U.space,V.space,args...)
  sparsities_1d = map(1:length(U.spaces_1d)) do d
    get_sparsity(U.spaces_1d[d],V.spaces_1d[d])
  end
  return TProductSparsity(sparsity,sparsities_1d)
end

function DofMaps.get_dof_map(V::TProductFESpace,args...)
  T = get_dof_eltype(V)
  dof_map = get_dof_map(V.space,args...)
  get_tp_dof_map(T,V.spaces_1d,dof_map)
end

function get_tp_dof_map(::Type{T},spaces_1d,dof_map) where T
  nnodes_1d = map(num_free_dofs,spaces_1d)
  reshape(dof_map,nnodes_1d...)
end

function get_tp_dof_map(::Type{T},spaces_1d,dof_map) where T<:MultiValue
  nnodes_1d = map(num_free_dofs,spaces_1d)
  ncomps = Int(length(dof_map)/prod(nnodes_1d))
  reshape(dof_map,nnodes_1d...,ncomps)
end

for F in (:(DofMaps.get_bg_dof_to_dof),:(DofMaps.get_dof_to_bg_dof))
  for T in (:SingleFieldFESpace,:FESpaceWithLinearConstraints)
    @eval begin
      function $F(bg_f::TProductFESpace,f::$T)
        $F(bg_f.space,f)
      end
    end
  end
end

get_tp_triangulation(f::TProductFESpace) = f.trian

"""
    get_tp_fe_basis(f::TProductFESpace) -> TProductFEBasis
"""
function get_tp_fe_basis(f::TProductFESpace)
  basis = map(get_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

"""
    get_tp_trial_fe_basis(f::TProductFESpace) -> TProductFEBasis
"""
function get_tp_trial_fe_basis(f::TProductFESpace)
  basis = map(get_trial_fe_basis,f.spaces_1d)
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

get_tp_trial_fe_basis(f::TrialFESpace{<:TProductFESpace}) = get_tp_trial_fe_basis(f.space)

# multi field

_remove_trial(f::SingleFieldFESpace) = _remove_trial(f.space)
_remove_trial(f::TProductFESpace) = f

function get_tp_triangulation(f::MultiFieldFESpace)
  s1 = _remove_trial(first(f.spaces))
  trian = get_tp_triangulation(s1)
  @check all(map(i->trian===get_tp_triangulation(_remove_trial(i)),f.spaces))
  trian
end

function get_tp_fe_basis(f::MultiFieldFESpace)
  D = length(_remove_trial(f[1]).spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> _remove_trial(sf).spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end

function get_tp_trial_fe_basis(f::MultiFieldFESpace)
  D = length(_remove_trial(f[1]).spaces_1d)
  basis = map(1:D) do d
    sfd = map(sf -> _remove_trial(sf).spaces_1d[d],f.spaces)
    mfd = MultiFieldFESpace(sfd)
    get_trial_fe_basis(mfd)
  end
  trian = get_tp_triangulation(f)
  TProductFEBasis(basis,trian)
end
