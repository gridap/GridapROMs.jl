struct EmbeddedFESpace{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  bg_space::SingleFieldFESpace
  fdof_to_bg_fdofs::AbstractVector
  ddof_to_bg_ddofs::AbstractVector
end

function EmbeddedFESpace(space::SingleFieldFESpace,bg_space::SingleFieldFESpace)
  fdof_to_bg_fdofs,ddof_to_bg_ddofs = get_active_dof_to_bg_dof(bg_space,space)
  EmbeddedFESpace(space,bg_space,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

FESpaces.ConstraintStyle(::Type{<:EmbeddedFESpace{S}}) where S = ConstraintStyle(S)

FESpaces.get_free_dof_ids(f::EmbeddedFESpace) = get_free_dof_ids(f.space)

FESpaces.get_triangulation(f::EmbeddedFESpace) = get_triangulation(f.space)

FESpaces.get_dof_value_type(f::EmbeddedFESpace) = get_dof_value_type(f.space)

FESpaces.get_cell_dof_ids(f::EmbeddedFESpace) = get_cell_dof_ids(f.space)

FESpaces.get_fe_basis(f::EmbeddedFESpace) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::EmbeddedFESpace) = get_trial_fe_basis(f.space)

FESpaces.get_fe_dof_basis(f::EmbeddedFESpace) = get_fe_dof_basis(f.space)

FESpaces.get_cell_isconstrained(f::EmbeddedFESpace) = get_cell_isconstrained(f.space)

FESpaces.get_cell_constraints(f::EmbeddedFESpace) = get_cell_constraints(f.space)

FESpaces.get_dirichlet_dof_ids(f::EmbeddedFESpace) = get_dirichlet_dof_ids(f.space)

FESpaces.get_cell_is_dirichlet(f::EmbeddedFESpace) = get_cell_is_dirichlet(f.space)

FESpaces.num_dirichlet_dofs(f::EmbeddedFESpace) = num_dirichlet_dofs(f.space)

FESpaces.num_dirichlet_tags(f::EmbeddedFESpace) = num_dirichlet_tags(f.space)

FESpaces.get_dirichlet_dof_tag(f::EmbeddedFESpace) = get_dirichlet_dof_tag(f.space)

FESpaces.get_vector_type(f::EmbeddedFESpace) = get_vector_type(f.space)

function FESpaces.scatter_free_and_dirichlet_values(f::EmbeddedFESpace,fv,dv)
  scatter_free_and_dirichlet_values(f.space,fv,dv)
end

function FESpaces.gather_free_and_dirichlet_values(f::EmbeddedFESpace,cv)
  FESpaces.gather_free_and_dirichlet_values(f.space,cv)
end

function FESpaces.gather_free_and_dirichlet_values!(fv,dv,f::EmbeddedFESpace,cv)
  gather_free_and_dirichlet_values!(fv,dv,f.space,cv)
end

# extended interface

get_emb_space(f::SingleFieldFESpace) = @abstractmethod
get_act_space(f::SingleFieldFESpace) = @abstractmethod
get_bg_space(f::SingleFieldFESpace) = @abstractmethod
get_active_fdof_to_bg_fdofs(f::SingleFieldFESpace) = @abstractmethod
get_active_ddof_to_bg_ddofs(f::SingleFieldFESpace) = @abstractmethod

get_emb_space(f::EmbeddedFESpace) = f
get_act_space(f::EmbeddedFESpace) = f.space
get_bg_space(f::EmbeddedFESpace) = f.bg_space
get_active_fdof_to_bg_fdofs(f::EmbeddedFESpace) = f.fdof_to_bg_fdofs
get_active_ddof_to_bg_ddofs(f::EmbeddedFESpace) = f.ddof_to_bg_ddofs

for F in (:get_emb_space,:get_act_space,:get_bg_space,:get_active_fdof_to_bg_fdofs,:get_active_ddof_to_bg_ddofs)
  for T in (:SingleFieldParamFESpace,:UnEvalTrialFESpace,:TransientTrialFESpace,:TrialFESpace)
    if !(F∈(:get_act_space,:get_bg_space) && T==:SingleFieldParamFESpace)
      @eval begin
        $F(f::$T) = $F(f.space)
      end
    end
  end
end

for F in (:get_act_space,:get_bg_space)
  @eval begin
    function $F(f::TrivialParamFESpace)
      TrivialParamFESpace($F(f.space),f.plength)
    end

    function $F(f::TrialParamFESpace)
      TrialParamFESpace($F(f.space),f.dirichlet_values)
    end
  end
end

zero_bg_free_values(f::SingleFieldFESpace) = zero_free_values(get_bg_space(f))
zero_bg_dirichlet_values(f::SingleFieldFESpace) = zero_dirichlet_values(get_bg_space(f))

function zero_bg_free_values(f::SingleFieldParamFESpace)
  bg_fv = zero_bg_free_values(get_emb_space(f))
  global_parameterize(bg_fv,param_length(f))
end

function zero_bg_dirichlet_values(f::SingleFieldParamFESpace)
  bg_dv = zero_bg_dirichlet_values(get_emb_space(f))
  global_parameterize(bg_dv,param_length(f))
end

function ExtendedFEFunction(f::SingleFieldFESpace,fv::AbstractVector,dv::AbstractVector)
  bg_cell_vals = scatter_extended_free_and_dirichlet_values(f,fv,dv)
  bg_cell_field = ExtendedCellField(f,bg_cell_vals)
  SingleFieldFEFunction(bg_cell_field,bg_cell_vals,fv,dv,f)
end

function ExtendedFEFunction(f::SingleFieldFESpace,fv::AbstractVector)
  dv = get_dirichlet_dof_values(f)
  ExtendedFEFunction(f,fv,dv)
end

function ExtendedFEFunction(
  f::EmbeddedFESpace{<:ZeroMeanFESpace},fv::AbstractVector,dv::AbstractVector
  )

  c = FESpaces._compute_new_fixedval(fv,dv,f.vol_i,f.vol,f.space.dof_to_fix)
  zmfv = lazy_map(+,fv,Fill(c,length(fv)))
  zmdv = dv .+ c
  FEFunction(f.space,zmfv,zmdv)
end

function extended_interpolate(object,f::SingleFieldFESpace)
  fv = zero_free_values(f)
  extended_interpolate!(object,fv,f)
end

function extended_interpolate!(object,fv,f::SingleFieldFESpace)
  interpolate!(object,fv,get_act_space(f))
  ExtendedFEFunction(f,fv)
end

function extended_interpolate_everywhere(object,f::SingleFieldFESpace)
  fv = zero_free_values(f)
  dv = zero_dirichlet_values(f)
  extended_interpolate_everywhere!(object,fv,dv,f)
end

function extended_interpolate_everywhere!(object,fv,dv,f::SingleFieldFESpace)
  interpolate_everywhere!(object,fv,dv,get_act_space(f))
  ExtendedFEFunction(f,fv,dv)
end

function extended_interpolate_dirichlet(object,f::SingleFieldFESpace)
  fv = zero_free_values(f)
  dv = zero_dirichlet_values(f)
  extended_interpolate_dirichlet!(object,fv,dv,f)
end

function extended_interpolate_dirichlet!(object,fv,dv,f::SingleFieldFESpace)
  interpolate_dirichlet!(object,fv,dv,get_act_space(f))
  ExtendedFEFunction(f,fv,dv)
end

function ExtendedCellField(f::SingleFieldFESpace,bg_cellvals)
  CellField(get_bg_space(f),bg_cellvals)
end

function scatter_extended_free_and_dirichlet_values(f::SingleFieldFESpace,fv,dv)
  bg_f = get_bg_space(f)
  bg_fv,bg_dv = extend_free_and_dirichlet_values(f,fv,dv)
  scatter_free_and_dirichlet_values(bg_f,bg_fv,bg_dv)
end

function gather_extended_dirichlet_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_dirichlet_values!(bg_dv,f,bg_cell_vals)
  dv
end

function gather_extended_dirichlet_values!(bg_dv,f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
  bg_dv
end

function gather_extended_free_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  gather_extended_free_values!(bg_fv,f,bg_cell_vals)
  bg_fv
end

function gather_extended_free_values!(bg_fv,f::SingleFieldFESpace,bg_cell_vals)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
  bg_fv
end

function gather_extended_free_and_dirichlet_values(f::SingleFieldFESpace,bg_cell_vals)
  bg_fv = zero_bg_free_values(f)
  bg_dv = zero_bg_dirichlet_values(f)
  gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f,bg_cell_vals)
end

function gather_extended_free_and_dirichlet_values!(bg_fv,bg_dv,f::SingleFieldFESpace,bg_cell_vals)
  gather_free_and_dirichlet_values!(bg_fv,bg_dv,get_bg_space(f),bg_cell_vals)
end

function extend_free_values(f::SingleFieldFESpace,fv)
  dv = zero_dirichlet_values(f)
  fv,dv = extend_free_and_dirichlet_values(f,fv,dv)
  return fv
end

function extend_dirichlet_values(f::SingleFieldFESpace,dv)
  fv = zero_free_values(f)
  fv,dv = extend_free_and_dirichlet_values(f,fv,dv)
  return dv
end

function extend_free_and_dirichlet_values(f::SingleFieldFESpace,fv,dv)
  bg_fv = zero_bg_free_values(f)
  bg_dv = zero_bg_dirichlet_values(f)
  _bg_vals_from_vals!(bg_fv,bg_dv,f,fv,dv)
  return bg_fv,bg_dv
end

# utils

function extend_cell_vals(f::SingleFieldFESpace,cell_vals)
  bg_f = get_bg_space(f)
  bg_fv,bg_dv = gather_extended_free_and_dirichlet_values(f,cell_vals)
  scatter_free_and_dirichlet_values(bg_f,bg_fv,bg_dv)
end

function _bg_vals_from_vals!(bg_fv,bg_dv,f::SingleFieldFESpace,fv,dv)
  _bg_vals_from_vals!(bg_fv,bg_dv,get_emb_space(f),fv,dv)
end

function _bg_vals_from_vals!(bg_fv,bg_dv,f::EmbeddedFESpace,fv,dv)
  for (fdof,bg_fdof) in enumerate(f.fdof_to_bg_fdofs)
    bg_fv[bg_fdof] = fv[fdof]
  end
  for (ddof,bg_ddof) in enumerate(f.ddof_to_bg_ddofs)
    bg_dv[bg_ddof] = dv[ddof]
  end
end

function _bg_vals_from_vals!(
  bg_fv::ConsecutiveParamVector,
  bg_dv::ConsecutiveParamVector,
  f::EmbeddedFESpace,
  fv::ConsecutiveParamVector,
  dv::ConsecutiveParamVector)

  bg_fdata = get_all_data(bg_fv)
  bg_ddata = get_all_data(bg_dv)
  fdata = get_all_data(fv)
  ddata = get_all_data(dv)

  for k in param_eachindex(bg_fv)
    for (fdof,bg_fdof) in enumerate(f.fdof_to_bg_fdofs)
      bg_fdata[bg_fdof,k] = fdata[fdof,k]
    end
    for (ddof,bg_ddof) in enumerate(f.ddof_to_bg_ddofs)
      bg_ddata[bg_ddof,k] = ddata[fdof,k]
    end
  end
end


function _bg_vals_from_vals!(
  bg_fv,
  bg_dv,
  f::EmbeddedFESpace{<:FESpaceWithLinearConstraints},
  fv,
  dv
  )

  T = eltype(bg_fv)

  for DOF in 1:length(f.space.DOF_to_mDOFs)
    pini = f.space.DOF_to_mDOFs.ptrs[DOF]
    pend = f.space.DOF_to_mDOFs.ptrs[DOF+1]-1
    val = zero(T)
    for p in pini:pend
      mDOF = f.space.DOF_to_mDOFs.data[p]
      coeff = f.space.DOF_to_coeffs.data[p]
      mdof = FESpaces._DOF_to_dof(mDOF,f.space.n_fmdofs)
      if mdof > 0
        val += fv[mdof]*coeff
      else
        val += dv[-mdof]*coeff
      end
    end
    dof = FESpaces._DOF_to_dof(DOF,f.space.n_fdofs)
    if dof > 0
      bg_fdof = f.fdof_to_bg_fdofs[dof]
      bg_fv[bg_fdof] = val
    else
      bg_ddof = f.ddof_to_bg_ddofs[-dof]
      bg_dv[-bg_ddof] = val
    end
  end
end

function _bg_vals_from_vals!(
  bg_fv::ConsecutiveParamVector,
  bg_dv::ConsecutiveParamVector,
  f::EmbeddedFESpace{<:FESpaceWithLinearConstraints},
  fv::ConsecutiveParamVector,
  dv::ConsecutiveParamVector
  )

  bg_fdata = get_all_data(bg_fv)
  bg_ddata = get_all_data(bg_dv)
  fdata = get_all_data(fv)
  ddata = get_all_data(dv)

  T = eltype(fdata)
  plength = param_length(fv)

  for DOF in 1:length(f.space.DOF_to_mDOFs)
    pini = f.space.DOF_to_mDOFs.ptrs[DOF]
    pend = f.space.DOF_to_mDOFs.ptrs[DOF+1]-1
    val = zeros(T,plength)
    for p in pini:pend
      mDOF = f.space.DOF_to_mDOFs.data[p]
      coeff = f.space.DOF_to_coeffs.data[p]
      mdof = FESpaces._DOF_to_dof(mDOF,f.space.n_fmdofs)
      for k in param_eachindex(bg_fv)
        if mdof > 0
          val[k] += fdata[mdof,k]*coeff
        else
          val[k] += ddata[-mdof,k]*coeff
        end
      end
    end
    dof = FESpaces._DOF_to_dof(DOF,f.space.n_fdofs)
    for k in param_eachindex(bg_fv)
      if dof > 0
        bg_fdof = f.fdof_to_bg_fdofs[dof]
        bg_fdata[bg_fdof,k] = val[k]
      else
        bg_ddof = f.ddof_to_bg_ddofs[-dof]
        bg_ddata[-bg_ddof,k] = val[k]
      end
    end
  end
end

struct BGCellDofIds{A<:AbstractArray,FI<:AbstractVector,FD<:AbstractVector} <: Map
  cell_dof_ids::A
  fdof_to_bg_fdofs::FI
  ddof_to_bg_ddofs::FD
end

function BGCellDofIds(cell_dof_ids::AbstractArray,fdof_to_bg_fdofs::AbstractArray)
  nddof_approx = maximum(fdof_to_bg_fdofs)
  ddof_to_bg_ddofs = IdentityVector(nddof_approx)
  BGCellDofIds(cell_dof_ids,fdof_to_bg_fdofs,ddof_to_bg_ddofs)
end

function Arrays.return_cache(k::BGCellDofIds,i::Int)
  cache = array_cache(k.cell_dof_ids)
  r = getindex!(cache,k.cell_dof_ids,i)
  (cache,CachedArray(r))
end

function Arrays.evaluate!(c,k::BGCellDofIds,i::Int)
  cache,a = c
  ids = getindex!(cache,k.cell_dof_ids,i)
  setsize!(a,size(ids))
  r = a.array
  for (j,idsj) in enumerate(ids)
    if idsj > 0
      r[j] = k.fdof_to_bg_fdofs[idsj]
    else
      r[j] = k.ddof_to_bg_ddofs[-idsj]
    end
  end
  return r
end

function get_bg_cell_dof_ids(f::EmbeddedFESpace,args...)
  cell_ids = get_cell_dof_ids(f,args...)
  k = BGCellDofIds(cell_ids,f.fdof_to_bg_fdofs,f.ddof_to_bg_ddofs)
  lazy_map(k,1:length(cell_ids))
end

function get_bg_cell_dof_ids(f::SingleFieldFESpace,args...)
  get_bg_cell_dof_ids(get_emb_space(f),args...)
end
