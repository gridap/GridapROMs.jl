function LexicographicFESpace(args...;kwargs...)
  LexicographicFESpace(FESpace(args...;kwargs...))
end

function LexicographicFESpace(f::FESpace)
  reindex_free_dof_ids(f,:coordinates;by=p->Tuple(p)[end:-1:1])
end

"""
    struct TProductFESpace{S} <: SingleFieldFESpace
      space::S
      spaces_1d::Vector{<:SingleFieldFESpace}
    end

A `SingleFieldFESpace` on a tensor product mesh, storing the D-dimensional
`space` and a vector of `D` 1D `spaces_1d`. Neither is reordered: their native
(Gridap-assigned) dof numbering is mapped to lexicographic rank wherever a 
tensor/Cartesian structure is needed (`get_dof_map`, `get_sparse_dof_map`).

All standard `SingleFieldFESpace` interface methods are delegated to `space`.

The preferred construction path is via [`TensorProductReferenceFE`](@ref):

```julia
model = TProductDiscreteModel((0,1,0,1),(10,10))
reffe = TensorProductReferenceFE(model,lagrangian,Float64,1)
V = FESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
```

Alternatively, the reffe tuple form is still supported:

```julia
Ω  = Triangulation(model)
dΩ = Measure(Ω,2)
V  = FESpace(Ω,(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")
```
"""
struct TProductFESpace{S} <: SingleFieldFESpace
  space::S
  spaces_1d::Vector{<:SingleFieldFESpace}
end

function FESpaces.FESpace(
  trian::TProductTriangulation,
  reffe::LagrangianRefFE;
  kwargs...
  )

  TProductFESpace(trian,reffe;kwargs...)
end

function FESpaces.FESpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...
  )

  TProductFESpace(trian,reffe;kwargs...)
end

function TProductFESpace(
  trian::TProductTriangulation,
  reffe::LagrangianRefFE;
  kwargs...
  )

  T = return_type(get_prebasis(reffe))
  order = get_order(reffe)
  TProductFESpace(trian,(lagrangian,(T,order),NamedTuple());kwargs...)
end

function TProductFESpace(
  trian::TProductTriangulation,
  reffe::Tuple{<:ReferenceFEName,Any,Any};
  kwargs...
  )

  model = get_background_model(trian)
  space = LexicographicFESpace(trian.trian,reffe;kwargs...)

  basis,reffe_args,reffe_kwargs = reffe
  T,order = reffe_args
  models_1d = univariate_models(model)
  cell_reffes_1d = map(models_1d) do model_1d
    ReferenceFE(model_1d,basis,eltype(T),order;reffe_kwargs...)
  end 
  spaces_1d = univariate_spaces(model,cell_reffes_1d;kwargs...)

  TProductFESpace(space,spaces_1d)
end

function univariate_spaces(
  model::DiscreteModel,
  cell_reffes;
  dirichlet_tags=Int[],
  dirichlet_masks=nothing,
  conformity=nothing,
  vector_type=nothing,
  kwargs...
  )

  if !isnothing(dirichlet_masks)
    for mask in dirichlet_masks
      !(all(mask) || !any(mask)) && _throw_tp_error()
    end
  end

  models_1d = univariate_models(model)
  diri_tags_1d = _get_1d_tags(model,dirichlet_tags)
  map(models_1d,cell_reffes,diri_tags_1d) do model,cell_reffe,tags
    LexicographicFESpace(model,cell_reffe;dirichlet_tags=tags,conformity,vector_type)
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

function DofMaps.get_sparsity(U::TProductFESpace,V::TProductFESpace)
  @check length(U.spaces_1d) == length(V.spaces_1d)
  sparsity = get_sparsity(U.space,V.space)
  sparsities_1d = map(1:length(U.spaces_1d)) do d
    get_sparsity(U.spaces_1d[d],V.spaces_1d[d])
  end
  return TProductSparsity(sparsity,sparsities_1d)
end

function DofMaps.get_sparsity(U::TProductFESpace,V::TProductFESpace,A::AbstractSparseMatrix)
  @check length(U.spaces_1d) == length(V.spaces_1d)
  sparsity = get_sparsity(U.space,V.space,A)
  sparsities_1d = map(1:length(U.spaces_1d)) do d
    get_sparsity(U.spaces_1d[d],V.spaces_1d[d])
  end
  return TProductSparsity(sparsity,sparsities_1d)
end

function DofMaps.get_sparse_dof_map(a::TProductSparsity,U::TProductFESpace,V::TProductFESpace)
  Tu = get_dof_eltype(U)
  Tv = get_dof_eltype(V)
  try
    full_ids = DofMaps.get_d_sparse_dofs_to_full_dofs(Tu,Tv,a)
    sparse_ids = sparsify_indices(full_ids)
    SparseMatrixDofMap(sparse_ids,full_ids,a)
  catch
    @warn "Could not build sparse tensor-product dof mapping. Must represent the
    jacobian using a linear dof map"
    get_sparse_dof_map(a.sparsity,U,V)
  end
end

function DofMaps.get_dof_map(V::TProductFESpace)
  T = get_dof_eltype(V)
  dof_map = get_dof_map(V.space)
  get_tp_dof_map(T,V.spaces_1d,dof_map)
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

for F in (:get_bg_dof_to_dof,:get_dof_to_bg_dof)
  for T in (:SingleFieldFESpace,:FESpaceWithLinearConstraints)
    @eval begin
      function DofMaps.$F(bg_f::TProductFESpace,f::$T)
        DofMaps.$F(bg_f.space,f)
      end
    end
  end
end