function get_dirichlet_cells end
get_dirichlet_cells(f::FESpace) = @abstractmethod
get_dirichlet_cells(f::UnconstrainedFESpace) = f.dirichlet_cells
get_dirichlet_cells(f::FESpaceWithConstantFixed) = get_dirichlet_cells(f.space)
get_dirichlet_cells(f::ZeroMeanFESpace) = get_dirichlet_cells(f.space)
get_dirichlet_cells(f::TrialFESpace) = get_dirichlet_cells(f.space)
get_dirichlet_cells(f::TProductFESpace) = get_dirichlet_cells(f.space)
get_dirichlet_cells(f::OrderedFESpace) = get_dirichlet_cells(f.space)

ParamDataStructures.param_length(f::FESpace) = 0

"""
    abstract type SingleFieldParamFESpace{S} <: SingleFieldFESpace end

Parametric extension of a `SingleFieldFESpace` in `Gridap`. The
FE spaces inhereting are (trial) spaces on which we can easily define a
`ParamFEFunction`.

Subtypes:
- [`TrivialParamFESpace`](@ref)
- [`TrialParamFESpace`](@ref)
"""
abstract type SingleFieldParamFESpace{S} <: SingleFieldFESpace end

FESpaces.ConstraintStyle(::Type{<:SingleFieldParamFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_fe_space(f::SingleFieldParamFESpace) = @abstractmethod

FESpaces.get_free_dof_ids(f::SingleFieldParamFESpace) = get_free_dof_ids(get_fe_space(f))

FESpaces.get_triangulation(f::SingleFieldParamFESpace) = get_triangulation(get_fe_space(f))

FESpaces.get_dof_value_type(f::SingleFieldParamFESpace) = get_dof_value_type(get_fe_space(f))

FESpaces.get_cell_dof_ids(f::SingleFieldParamFESpace) = get_cell_dof_ids(get_fe_space(f))

FESpaces.get_fe_basis(f::SingleFieldParamFESpace) = get_fe_basis(get_fe_space(f))

FESpaces.get_trial_fe_basis(f::SingleFieldParamFESpace) = get_trial_fe_basis(get_fe_space(f))

FESpaces.get_fe_dof_basis(f::SingleFieldParamFESpace) = get_fe_dof_basis(get_fe_space(f))

FESpaces.get_cell_isconstrained(f::SingleFieldParamFESpace) = get_cell_isconstrained(get_fe_space(f))

FESpaces.get_cell_constraints(f::SingleFieldParamFESpace) = get_cell_constraints(get_fe_space(f))

FESpaces.get_dirichlet_dof_ids(f::SingleFieldParamFESpace) = get_dirichlet_dof_ids(get_fe_space(f))

FESpaces.get_cell_is_dirichlet(f::SingleFieldParamFESpace) = get_cell_is_dirichlet(get_fe_space(f))

FESpaces.num_dirichlet_dofs(f::SingleFieldParamFESpace) = num_dirichlet_dofs(get_fe_space(f))

FESpaces.num_dirichlet_tags(f::SingleFieldParamFESpace) = num_dirichlet_tags(get_fe_space(f))

FESpaces.get_dirichlet_dof_tag(f::SingleFieldParamFESpace) = get_dirichlet_dof_tag(get_fe_space(f))

function FESpaces.scatter_free_and_dirichlet_values(f::SingleFieldParamFESpace,fv,dv)
  scatter_free_and_dirichlet_values(get_fe_space(f),fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::SingleFieldParamFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,get_fe_space(f),cv)
end

function FESpaces.gather_free_and_dirichlet_values(f::SingleFieldParamFESpace,cv)
  fv = zero_free_values(f)
  dv = zero_dirichlet_values(f)
  gather_free_and_dirichlet_values!(fv,dv,f,cv)
end

function DofMaps.get_dof_map(f::SingleFieldParamFESpace,args...)
  get_dof_map(get_fe_space(f),args...)
end

function DofMaps.get_sparse_dof_map(f::SingleFieldParamFESpace,g::SingleFieldFESpace,args...)
  get_sparse_dof_map(get_fe_space(f),g,args...)
end

get_dirichlet_cells(f::SingleFieldParamFESpace) = get_dirichlet_cells(get_fe_space(f))

# These functions allow us to use global ParamArrays

"""
    get_vector_type2(f::SingleFieldParamFESpace) -> Type

Returns the vector type of the underlying un-parametric FESpace contained in `f`
"""
get_vector_type2(f::FESpace) = @abstractmethod
get_vector_type2(f::SingleFieldParamFESpace) = get_vector_type(get_fe_space(f))

"""
    param_zero_free_values(f::FESpace) -> AbstractParamVector

Parametric version of the function `zero_free_values` in `Gridap`
"""
function param_zero_free_values(f::FESpace)
  V = get_vector_type2(f)
  L = param_length(f)
  v = allocate_vector(V,get_free_dof_ids(f))
  fill!(v,zero(eltype(V)))
  pv = global_parameterize(v,L)
  return pv
end

"""
    param_zero_dirichlet_values(f::FESpace) -> AbstractParamVector

Parametric version of the function `zero_dirichlet_values` in `Gridap`
"""
function param_zero_dirichlet_values(f::FESpace)
  V = get_vector_type2(f)
  L = param_length(f)
  v = allocate_vector(V,get_dirichlet_dof_ids(f))
  fill!(v,zero(eltype(V)))
  pv = global_parameterize(v,L)
  return pv
end

function FESpaces.get_vector_type(f::SingleFieldParamFESpace)
  V = get_vector_type(get_fe_space(f))
  L = param_length(f)
  PV = global_parameterize(V(),L)
  typeof(PV)
end

function FESpaces.zero_free_values(f::SingleFieldParamFESpace)
  param_zero_free_values(f)
end

function FESpaces.zero_dirichlet_values(f::SingleFieldParamFESpace)
  param_zero_dirichlet_values(f)
end

function FESpaces.FEFunction(
  pf::SingleFieldParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  f = get_fe_space(pf)
  pf′ = remove_layer(pf)
  c = FESpaces._compute_new_fixedval(
    free_values,
    dirichlet_values,
    f.vol_i,
    f.vol,
    f.space.dof_to_fix
  )
  fv = free_values + c
  dv = dirichlet_values + c
  FEFunction(pf′,fv,dv)
end

function FESpaces.EvaluationFunction(
  pf::SingleFieldParamFESpace{<:ZeroMeanFESpace},
  free_values::AbstractParamVector)

  pf′ = remove_layer(pf)
  FEFunction(pf′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceWithConstantFixed{FESpaces.FixConstant},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  )

  @assert innerlength(dv) == 1
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryInserted(fv,f.dof_to_fix,get_param_entry(dv,1))
  scatter_free_and_dirichlet_values(f.space,_fv,_dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceWithConstantFixed{FESpaces.DoNotFixConstant},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  )

  @assert innerlength(dv) == 0
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(
  pf::SingleFieldParamFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  f = get_fe_space(pf)
  _fv,_dv = zero_free_and_dirichlet_values(f.space)
  @assert length(_dv) == 0
  pfv = global_parameterize(_fv,param_length(pf))
  pdv = global_parameterize(_dv,param_length(pf))
  gather_free_and_dirichlet_values!(pfv,pdv,f.space,cv)

  fv = ParamVectorWithEntryRemoved(pfv,f.dof_to_fix)
  dv = get_param_entry(pfv,f.dof_to_fix:f.dof_to_fix)
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv::AbstractParamVector,
  dv::AbstractParamVector,
  f::FESpaceWithConstantFixed{T},
  cv) where T<:FESpaces.FixConstant

  @assert innerlength(dv) == 1
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryInserted(fv,f.dof_to_fix,zeros(eltype2(fv),param_length(fv)))
  gather_free_and_dirichlet_values!(_fv,_dv,f,cv)
  dv.data[1,:] = _fv.value
  (fv,dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceWithLinearConstraints,
  fmdof_to_val::AbstractParamVector,
  dmdof_to_val::AbstractParamVector)

  @check param_length(fmdof_to_val) == param_length(dmdof_to_val)
  plength = param_length(fmdof_to_val)
  fdof_to_val = global_parameterize(zero_free_values(f.space),plength)
  ddof_to_val = global_parameterize(zero_dirichlet_values(f.space),plength)

  FESpaces._setup_dof_to_val!(
    fdof_to_val,
    ddof_to_val,
    fmdof_to_val,
    dmdof_to_val,
    f.DOF_to_mDOFs,
    f.DOF_to_coeffs,
    f.n_fdofs,
    f.n_fmdofs)

  scatter_free_and_dirichlet_values(f.space,fdof_to_val,ddof_to_val)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val::AbstractParamVector,
  dmdof_to_val::AbstractParamVector,
  f::FESpaceWithLinearConstraints,
  cell_to_ludof_to_val)

  @check param_length(fmdof_to_val) == param_length(dmdof_to_val)
  plength = param_length(fmdof_to_val)

  _fv,_dv = zero_free_and_dirichlet_values(f.space)
  fdof_to_val = global_parameterize(_fv,plength)
  ddof_to_val = global_parameterize(_dv,plength)
  gather_free_and_dirichlet_values!(fdof_to_val,ddof_to_val,f.space,cell_to_ludof_to_val)

  FESpaces._setup_mdof_to_val!(
    fmdof_to_val,
    dmdof_to_val,
    fdof_to_val,
    ddof_to_val,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)

  fmdof_to_val,dmdof_to_val
end

function FESpaces._fill_dirichlet_values_for_tag!(
  dirichlet_values::AbstractParamVector,
  dv::AbstractParamVector,
  tag,
  dirichlet_dof_to_tag)

  @check param_length(dirichlet_values) == param_length(dv)
  diri_data = get_all_data(dirichlet_values)
  dv_data = get_all_data(dv)
  for dof in 1:innerlength(dv)
    if dirichlet_dof_to_tag[dof] == tag
      @inbounds for k in param_eachindex(dv)
        diri_data[dof,k] = dv_data[dof,k]
      end
    end
  end
end

function FESpaces._free_and_dirichlet_values_fill!(
  free_vals::AbstractParamVector,
  dirichlet_vals::AbstractParamVector,
  cache_vals,
  cache_dofs,
  cell_vals,
  cell_dofs,
  cells)

  @check param_length(free_vals) == param_length(dirichlet_vals)
  free_data = get_all_data(free_vals)
  diri_data = get_all_data(dirichlet_vals)
  for cell in cells
    vals = getindex!(cache_vals,cell_vals,cell)
    dofs = getindex!(cache_dofs,cell_dofs,cell)
    for k in param_eachindex(free_vals)
      val = param_getindex(vals,k)
      for (i,dof) in enumerate(dofs)
        if dof > 0
          free_data[dof,k] = val[i]
        elseif dof < 0
          diri_data[-dof,k] = val[i]
        else
          @unreachable "dof ids either positive or negative, not zero"
        end
      end
    end
  end
end

function FESpaces._compute_new_fixedval(
  fv::AbstractParamVector,
  dv::AbstractParamVector,
  vol_i,
  vol,
  fixed_dof)

  @assert innerlength(fv) + 1 == length(vol_i)
  @assert innerlength(dv) == 1
  @assert param_length(fv) == param_length(dv)

  c = zeros(eltype(vol_i),param_length(fv))
  @inbounds for k in param_eachindex(fv)
    ck = c[k]
    fvk = fv[k]
    dvk = dv[k]
    for i=1:fixed_dof-1
      ck += fvk[i]*vol_i[i]
    end
    ck += first(dvk)*vol_i[fixed_dof]
    for i=fixed_dof+1:length(vol_i)
      ck += fvk[i-1]*vol_i[i]
    end
    ck = -ck/vol
    c[k] = ck
  end

  return c
end

function FESpaces._setup_dof_to_val!(
  fdof_to_val::ConsecutiveParamVector,
  ddof_to_val::ConsecutiveParamVector,
  fmdof_to_val::ConsecutiveParamVector,
  dmdof_to_val::ConsecutiveParamVector,
  DOF_to_mDOFs,
  DOF_to_coeffs,
  n_fdofs,
  n_fmdofs)

  @check (param_length(fdof_to_val) == param_length(ddof_to_val) ==
          param_length(fmdof_to_val) == param_length(dmdof_to_val))
  f2v = get_all_data(fdof_to_val)
  d2v = get_all_data(ddof_to_val)
  fm2v = get_all_data(fmdof_to_val)
  dm2v = get_all_data(dmdof_to_val)

  T = eltype2(fdof_to_val)
  plength = param_length(fdof_to_val)
  val = zeros(T,plength)

  for DOF in 1:length(DOF_to_mDOFs)
    pini = DOF_to_mDOFs.ptrs[DOF]
    pend = DOF_to_mDOFs.ptrs[DOF+1]-1
    fill!(val,zero(T))
    for p in pini:pend
      mDOF = DOF_to_mDOFs.data[p]
      coeff = DOF_to_coeffs.data[p]
      mdof = FESpaces._DOF_to_dof(mDOF,n_fmdofs)
      if mdof > 0
        fmdof = mdof
        @inbounds for k in 1:plength
          val[k] += fm2v[fmdof,k]*coeff
        end
      else
        dmdof = -mdof
        @inbounds for k in 1:plength
          val[k] += dm2v[dmdof,k]*coeff
        end
      end
    end
    dof = FESpaces._DOF_to_dof(DOF,n_fdofs)
    if dof > 0
      fdof = dof
      @inbounds for k in 1:plength
        f2v[fdof,k] = val[k]
      end
    else
      ddof = -dof
      @inbounds for k in 1:plength
        d2v[ddof,k] = val[k]
      end
    end
  end
end

function FESpaces._setup_mdof_to_val!(
  fmdof_to_val::ConsecutiveParamVector,
  dmdof_to_val::ConsecutiveParamVector,
  fdof_to_val::ConsecutiveParamVector,
  ddof_to_val::ConsecutiveParamVector,
  mDOF_to_DOF,
  n_fdofs,
  n_fmdofs)

  @check (param_length(fdof_to_val) == param_length(ddof_to_val) ==
          param_length(fmdof_to_val) == param_length(dmdof_to_val))
  f2v = get_all_data(fdof_to_val)
  d2v = get_all_data(ddof_to_val)
  fm2v = get_all_data(fmdof_to_val)
  dm2v = get_all_data(dmdof_to_val)

  T = eltype2(fdof_to_val)
  plength = param_length(fdof_to_val)
  val = zeros(T,plength)

  for mDOF in 1:length(mDOF_to_DOF)
    DOF = mDOF_to_DOF[mDOF]
    dof = FESpaces._DOF_to_dof(DOF,n_fdofs)
    if dof > 0
      fdof = dof
      @inbounds for k in 1:plength
        val[k] = f2v[fdof,k]
      end
    else
      ddof = -dof
      @inbounds for k in 1:plength
        val[k] = d2v[ddof,k]
      end
    end
    mdof = FESpaces._DOF_to_dof(mDOF,n_fmdofs)
    if mdof > 0
      fmdof = mdof
      @inbounds for k in 1:plength
        fm2v[fmdof,k] = val[k]
      end
    else
      dmdof = -mdof
      @inbounds for k in 1:plength
        dm2v[dmdof,k] = val[k]
      end
    end
  end

end

# ordered spaces

get_fe_space2(pf::SingleFieldParamFESpace{<:OrderedFESpace}) = get_fe_space(get_fe_space(pf))

function FESpaces.FEFunction(
  pf::SingleFieldParamFESpace{<:OrderedFESpace{<:ZeroMeanFESpace}},
  free_values::AbstractParamVector,
  dirichlet_values::AbstractParamVector)

  f = get_fe_space2(pf)
  pf′ = remove_layer(pf)
  c = FESpaces._compute_new_fixedval(
    free_values,
    dirichlet_values,
    f.vol_i,
    f.vol,
    f.space.dof_to_fix
  )
  fv = free_values + c
  dv = dirichlet_values + c
  FEFunction(pf′,fv,dv)
end

function FESpaces.EvaluationFunction(
  pf::SingleFieldParamFESpace{<:OrderedFESpace{<:ZeroMeanFESpace}},
  free_values::AbstractParamVector)

  pf′ = remove_layer(pf)
  FEFunction(pf′,free_values)
end

function FESpaces.scatter_free_and_dirichlet_values(
  of::OrderedFESpace{<:FESpaceWithConstantFixed{FESpaces.FixConstant}},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  )

  @assert innerlength(dv) == 1
  f = get_fe_space(of)
  of′ = remove_layer(of)
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryInserted(fv,f.dof_to_fix,get_param_entry(dv,1))
  scatter_free_and_dirichlet_values(of′,_fv,_dv)
end

function FESpaces.scatter_free_and_dirichlet_values(
  of::OrderedFESpace{<:FESpaceWithConstantFixed{FESpaces.DoNotFixConstant}},
  fv::AbstractParamVector,
  dv::AbstractParamVector
  )

  @assert innerlength(dv) == 0
  of′ = remove_layer(of)
  scatter_free_and_dirichlet_values(of′,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(
  pf::SingleFieldParamFESpace{<:OrderedFESpace{<:FESpaceWithConstantFixed{T}}},
  cv) where T<:FESpaces.FixConstant

  f = get_fe_space(pf)
  _fv,_dv = zero_free_and_dirichlet_values(f.space)
  @assert length(_dv) == 0
  pfv = global_parameterize(_fv,param_length(pf))
  pdv = global_parameterize(_dv,param_length(pf))
  gather_free_and_dirichlet_values!(pfv,pdv,f.space,cv)

  fv = ParamVectorWithEntryRemoved(pfv,f.dof_to_fix)
  dv = get_param_entry(pfv,f.dof_to_fix:f.dof_to_fix)
  (fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values!(
  fv::AbstractParamVector,
  dv::AbstractParamVector,
  f::OrderedFESpace{<:FESpaceWithConstantFixed{T}},
  cv) where T<:FESpaces.FixConstant

  @assert innerlength(dv) == 1
  _dv = similar(dv,eltype(dv),0)
  _fv = ParamVectorWithEntryInserted(fv,f.dof_to_fix,zeros(eltype2(fv),param_length(fv)))
  gather_free_and_dirichlet_values!(_fv,_dv,f,cv)
  dv.data[1,:] = _fv.value
  (fv,dv)
end

# utils

function zero_free_and_dirichlet_values(f::SingleFieldFESpace)
  fv = zero_free_values(f)
  dv = zero_dirichlet_values(f)
  (fv,dv)
end

remove_layer(f::SingleFieldFESpace) = @abstractmethod
remove_layer(f::SingleFieldParamFESpace{<:UnconstrainedFESpace}) = f
remove_layer(f::SingleFieldParamFESpace{<:OrderedFESpace{<:UnconstrainedFESpace}}) = f
remove_layer(f::OrderedFESpace) = OrderedFESpace(f.space.space,f.cell_odofs_ids,f.bg_odofs_to_act_odofs)
