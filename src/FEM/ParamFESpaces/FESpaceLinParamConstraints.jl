struct FESpaceLinParamConstraints{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  n_fdofs::Int
  n_fmdofs::Int
  mDOF_to_DOF::Vector
  DOF_to_mDOFs::Table
  DOF_to_coeffs::BidimensionalTable
  cell_to_lmdof_to_mdof::Table
  cell_to_ldof_to_dof::Table
end

function FESpaces.FESpaceWithLinearConstraints(
  space::SingleFieldFESpace,
  n_fdofs::Int,
  n_fmdofs::Int,
  mDOF_to_DOF::Vector,
  DOF_to_mDOFs::Table,
  DOF_to_coeffs::BidimensionalTable,
  cell_to_lmdof_to_mdof::Table,
  cell_to_ldof_to_dof::Table)

  FESpaceLinParamConstraints(
    space,
    n_fdofs,
    n_fmdofs,
    mDOF_to_DOF,
    DOF_to_mDOFs,
    DOF_to_coeffs,
    cell_to_lmdof_to_mdof,
    cell_to_ldof_to_dof)
end

function FESpaces.FESpaceWithLinearConstraints(
  sDOF_to_dof::AbstractVector{<:Integer},
  sDOF_to_dofs::Table,
  sDOF_to_coeffs::BidimensionalTable,
  space::SingleFieldFESpace)

  n_fdofs = num_free_dofs(space)
  n_ddofs = num_dirichlet_dofs(space)
  n_DOFs = n_fdofs + n_ddofs

  DOF_to_DOFs,DOF_to_coeffs = FESpaces._prepare_DOF_to_DOFs(
    sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs,n_fdofs,n_DOFs)

  FESpaces.FESpaceWithLinearConstraints!(DOF_to_DOFs,DOF_to_coeffs,space)
end

function FESpaces.FESpaceWithLinearConstraints(
  fdof_to_dofs::Table,
  fdof_to_coeffs::BidimensionalTable,
  ddof_to_dofs::Table,
  ddof_to_coeffs::BidimensionalTable,
  space::SingleFieldFESpace)

  DOF_to_DOFs,DOF_to_coeffs = FESpaces._merge_free_and_diri_constraints(
    fdof_to_dofs,fdof_to_coeffs,ddof_to_dofs,ddof_to_coeffs)
  FESpaces.FESpaceWithLinearConstraints!(DOF_to_DOFs,DOF_to_coeffs,space)
end

function FESpaces.FESpaceWithLinearConstraints(
  DOF_to_DOFs::Table,DOF_to_coeffs::BidimensionalTable,space::SingleFieldFESpace)
  FESpaces.FESpaceWithLinearConstraints!(copy(DOF_to_DOFs),copy(DOF_to_coeffs),space::SingleFieldFESpace)
end

function FESpaces.FESpaceWithLinearConstraints!(
  DOF_to_DOFs::Table,DOF_to_coeffs::BidimensionalTable,space::SingleFieldFESpace)

  n_fdofs = num_free_dofs(space)
  mDOF_to_DOF,n_fmdofs = FESpaces._find_master_dofs(DOF_to_DOFs,n_fdofs)
  DOF_to_mDOFs = FESpaces._renumber_constraints!(DOF_to_DOFs,mDOF_to_DOF)
  cell_to_ldof_to_dof = Table(get_cell_dof_ids(space))
  cell_to_lmdof_to_mdof = FESpaces._setup_cell_to_lmdof_to_mdof(cell_to_ldof_to_dof,DOF_to_mDOFs,n_fdofs,n_fmdofs)

  FESpaceLinParamConstraints(
    space,
    n_fdofs,
    n_fmdofs,
    mDOF_to_DOF,
    DOF_to_mDOFs,
    DOF_to_coeffs,
    cell_to_lmdof_to_mdof,
    cell_to_ldof_to_dof)

end

FESpaces.get_triangulation(f::FESpaceLinParamConstraints) = get_triangulation(f.space)

FESpaces.get_cell_dof_ids(f::FESpaceLinParamConstraints) = f.cell_to_lmdof_to_mdof

FESpaces.get_fe_dof_basis(f::FESpaceLinParamConstraints) = get_fe_dof_basis(f.space)

FESpaces.get_dirichlet_dof_ids(f::FESpaceLinParamConstraints) = Base.OneTo(length(f.mDOF_to_DOF) - f.n_fmdofs)

FESpaces.num_dirichlet_tags(f::FESpaceLinParamConstraints) = num_dirichlet_tags(f.space)

FESpaces.get_free_dof_ids(f::FESpaceLinParamConstraints) = Base.OneTo(f.n_fmdofs)

FESpaces.ConstraintStyle(::Type{<:FESpaceLinParamConstraints}) = Constrained()

FESpaces.get_vector_type(f::FESpaceLinParamConstraints) = get_vector_type(f.space)

FESpaces.get_fe_basis(f::FESpaceLinParamConstraints) = get_fe_basis(f.space)

FESpaces.get_trial_fe_basis(f::FESpaceLinParamConstraints) = get_trial_fe_basis(f.space)

function FESpaces.get_cell_isconstrained(f::FESpaceLinParamConstraints)
  n = length(get_cell_dof_ids(f))
  Fill(true,n)
end

function FESpaces.get_dirichlet_dof_tag(f::FESpaceLinParamConstraints)
  ddof_to_tag = get_dirichlet_dof_tag(f.space)
  dmdof_to_tag = zeros(eltype(ddof_to_tag),num_dirichlet_dofs(f))
  FESpaces._setup_ddof_to_tag!(
    dmdof_to_tag,
    ddof_to_tag,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)
  dmdof_to_tag
end

function FESpaces.get_dirichlet_dof_values(f::FESpaceLinParamConstraints)
  ddof_to_tag = get_dirichlet_dof_values(f.space)
  dmdof_to_tag = zeros(eltype(ddof_to_tag),num_dirichlet_dofs(f))
  FESpaces._setup_ddof_to_tag!(
    dmdof_to_tag,
    ddof_to_tag,
    f.mDOF_to_DOF,
    f.n_fdofs,
    f.n_fmdofs)
  dmdof_to_tag
end

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceLinParamConstraints,
  fmdof_to_val,
  dmdof_to_val
  )

  fdof_to_val = zero_free_values(f.space)
  ddof_to_val = zero_dirichlet_values(f.space)
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

function FESpaces.scatter_free_and_dirichlet_values(
  f::FESpaceLinParamConstraints,
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

function FESpaces.gather_free_and_dirichlet_values(
  f::FESpaceLinParamConstraints,
  cell_to_ludof_to_val
  )

  fdof_to_val,ddof_to_val = FESpaces.gather_free_and_dirichlet_values(f.space,cell_to_ludof_to_val)
  fmdof_to_val = zero_free_values(f)
  dmdof_to_val = zero_dirichlet_values(f)
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

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val,
  dmdof_to_val,
  f::FESpaceLinParamConstraints,
  cell_to_ludof_to_val
  )

  fdof_to_val,ddof_to_val = FESpaces.gather_free_and_dirichlet_values(f.space,cell_to_ludof_to_val)
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

function FESpaces.gather_free_and_dirichlet_values!(
  fmdof_to_val::AbstractParamVector,
  dmdof_to_val::AbstractParamVector,
  f::FESpaceLinParamConstraints,
  cell_to_ludof_to_val
  )

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

function CellData.CellField(f::FESpaceLinParamConstraints,cellvals)
  CellField(f.space,cellvals)
end

function FESpaces.get_cell_constraints(f::FESpaceLinParamConstraints)
  k = FESpaces.LinearConstraintsMap(
    f.DOF_to_mDOFs,
    f.DOF_to_coeffs,
    length(f.mDOF_to_DOF),
    f.n_fmdofs,
    f.n_fdofs)

  cell_to_mat = get_cell_constraints(f.space)
  lazy_map(k,f.cell_to_lmdof_to_mdof,f.cell_to_ldof_to_dof,cell_to_mat)
end

const LinearParamConstraintsMap{A,B<:BidimensionalTable} = FESpaces.LinearConstraintsMap{A,B}

ParamDataStructures.param_length(a::LinearParamConstraintsMap) = param_length(a.DOF_to_coeffs)

function ParamDataStructures.param_getindex(k::LinearParamConstraintsMap,i::Int)
  FESpaces.LinearConstraintsMap(
    k.DOF_to_mDOFs,
    param_getindex(k.DOF_to_coeffs,i),
    k.n_mDOFs,
    k.n_fmdofs,
    k.n_fdofs)
end

Arrays.testitem(a::LinearParamConstraintsMap) = param_getindex(a,1)

function Arrays.return_cache(
  k::LinearParamConstraintsMap,lmdof_to_mdof,ldof_to_dof,mat
  )

  ki = testitem(k)
  ci = return_cache(ki,lmdof_to_mdof,ldof_to_dof,mat)
  vi = evaluate!(ci,ki,lmdof_to_mdof,ldof_to_dof,mat)
  data = Vector{typeof(vi)}(undef,param_length(k))
  c = Vector{typeof(ci)}(undef,param_length(k))
  for i in 1:param_length(k)
    c[i] = return_cache(param_getindex(k,i),lmdof_to_mdof,ldof_to_dof,mat)
  end
  GenericParamBlock(data),c
end

function Arrays.evaluate!(
  cache,k::LinearParamConstraintsMap,lmdof_to_mdof,ldof_to_dof,mat
  )

  r,c = cache
  for i in 1:param_length(k)
    r.data[i] = evaluate!(c[i],param_getindex(k,i),lmdof_to_mdof,ldof_to_dof,mat)
  end
  r
end

# utils

function FESpaces._merge_free_and_diri_constraints(
  fdof_to_dofs,fdof_to_coeffs::BidimensionalTable,ddof_to_dofs,ddof_to_coeffs::BidimensionalTable)
  n_fdofs = length(fdof_to_dofs)
  DOF_to_DOFs = append_tables_globally(fdof_to_dofs,ddof_to_dofs)
  for i in 1:length(DOF_to_DOFs.data)
    dof = DOF_to_DOFs.data[i]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    DOF_to_DOFs.data[i] = DOF
  end
  DOF_to_coeffs = BidimensionalTable(vcat(fdof_to_coeffs.data,ddof_to_coeffs.data),DOF_to_DOFs.ptrs)
  DOF_to_DOFs,DOF_to_coeffs
end

function FESpaces._prepare_DOF_to_DOFs(
  sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs::BidimensionalTable,n_fdofs,n_DOFs)

  Tp = eltype(sDOF_to_dofs.ptrs)
  Td = eltype(sDOF_to_dofs.data)
  Tc = eltype(sDOF_to_coeffs.data)

  DOF_to_DOFs_ptrs = ones(Tp,n_DOFs+1)

  n_sDOFs = length(sDOF_to_dof)

  for sDOF in 1:n_sDOFs
    a = sDOF_to_dofs.ptrs[sDOF]
    b = sDOF_to_dofs.ptrs[sDOF+1]
    dof = sDOF_to_dof[sDOF]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    DOF_to_DOFs_ptrs[DOF+1] = b-a
  end

  length_to_ptrs!(DOF_to_DOFs_ptrs)
  ndata = DOF_to_DOFs_ptrs[end]-1
  plength = size(sDOF_to_coeffs.data,2)
  DOF_to_DOFs_data = zeros(Td,ndata)
  DOF_to_coeffs_data = ones(Tc,ndata,plength)

  for DOF in 1:n_DOFs
    q = DOF_to_DOFs_ptrs[DOF]
    DOF_to_DOFs_data[q] = DOF
  end

  for sDOF in 1:n_sDOFs
    dof = sDOF_to_dof[sDOF]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    q = DOF_to_DOFs_ptrs[DOF]-1
    pini = sDOF_to_dofs.ptrs[sDOF]
    pend = sDOF_to_dofs.ptrs[sDOF+1]-1
    for (i,p) in enumerate(pini:pend)
      _dof = sDOF_to_dofs.data[p]
      _DOF = FESpaces._dof_to_DOF(_dof,n_fdofs)
      DOF_to_DOFs_data[q+i] = _DOF
      for l in axes(sDOF_to_coeffs.data,2)
        coeff = sDOF_to_coeffs.data[p,l]
        DOF_to_coeffs_data[q+i,l] = coeff
      end
    end
  end

  DOF_to_DOFs = Table(DOF_to_DOFs_data,DOF_to_DOFs_ptrs)
  DOF_to_coeffs = BidimensionalTable(DOF_to_coeffs_data,DOF_to_DOFs_ptrs)

  DOF_to_DOFs,DOF_to_coeffs
end

function FESpaces._setup_dof_to_val!(
  fdof_to_val::ConsecutiveParamVector,
  ddof_to_val::ConsecutiveParamVector,
  fmdof_to_val::ConsecutiveParamVector,
  dmdof_to_val::ConsecutiveParamVector,
  DOF_to_mDOFs,
  DOF_to_coeffs::BidimensionalTable,
  n_fdofs,
  n_fmdofs
  )

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
      mdof = FESpaces._DOF_to_dof(mDOF,n_fmdofs)
      if mdof > 0
        fmdof = mdof
        @inbounds for k in 1:plength
          val[k] += fm2v[fmdof,k]*DOF_to_coeffs.data[p,k]
        end
      else
        dmdof = -mdof
        @inbounds for k in 1:plength
          val[k] += dm2v[dmdof,k]*DOF_to_coeffs.data[p,k]
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

function AgFEM._setup_agfem_constraints(
  n_fdofs,
  acell_to_acellin,
  acell_to_dof_ids,
  acell_to_coeffs::AbstractVector{<:ParamBlock},
  acell_to_proj,
  acell_to_gcell)

  n_acells = length(acell_to_acellin)
  fdof_to_isagg = fill(true,n_fdofs)
  fdof_to_acell = zeros(Int32,n_fdofs)
  fdof_to_ldof = zeros(Int16,n_fdofs)
  cache = array_cache(acell_to_dof_ids)
  for acell in 1:n_acells
    acellin = acell_to_acellin[acell]
    iscut = acell != acellin
    dofs = getindex!(cache,acell_to_dof_ids,acell)
    gcell = acell_to_gcell[acell]
    for (ldof,dof) in enumerate(dofs)
      if dof > 0
        fdof = dof
        acell_dof = fdof_to_acell[fdof]
        fdof_to_isagg[fdof] &= iscut
        if acell_dof == 0 || gcell > acell_to_gcell[acell_dof]
          fdof_to_acell[fdof] = acell
          fdof_to_ldof[fdof] = ldof
         end
      end
    end
  end

  aggdof_to_fdof = findall(fdof_to_isagg)

  n_aggdofs = length(aggdof_to_fdof)
  aggdof_to_dofs_ptrs = zeros(Int32,n_aggdofs+1)

  for aggdof in 1:n_aggdofs
    fdof = aggdof_to_fdof[aggdof]
    acell = fdof_to_acell[fdof]
    acellin = acell_to_acellin[acell]
    dofs = getindex!(cache,acell_to_dof_ids,acellin)
    aggdof_to_dofs_ptrs[aggdof+1] = length(dofs)
  end

  length_to_ptrs!(aggdof_to_dofs_ptrs)
  ndata = aggdof_to_dofs_ptrs[end]-1
  aggdof_to_dofs_data = zeros(Int,ndata)

  for aggdof in 1:n_aggdofs
    fdof = aggdof_to_fdof[aggdof]
    acell = fdof_to_acell[fdof]
    acellin = acell_to_acellin[acell]
    dofs = getindex!(cache,acell_to_dof_ids,acellin)
    p = aggdof_to_dofs_ptrs[aggdof]-1
    for (i,dof) in enumerate(dofs)
      aggdof_to_dofs_data[p+i] = dof
    end
  end

  aggdof_to_dofs = Table(aggdof_to_dofs_data,aggdof_to_dofs_ptrs)

  cache2 = array_cache(acell_to_coeffs)
  cache3 = array_cache(acell_to_proj)

  plength = param_length(testitem(acell_to_coeffs))
  T = eltype2(eltype(acell_to_coeffs))
  z = zero(T)

  aggdof_to_coeffs_data = zeros(T,ndata,plength)
  for aggdof in 1:n_aggdofs
    fdof = aggdof_to_fdof[aggdof]
    acell = fdof_to_acell[fdof]
    coeffs = getindex!(cache2,acell_to_coeffs,acell)
    proj = getindex!(cache3,acell_to_proj,acell)
    ldof = fdof_to_ldof[fdof]
    p = aggdof_to_dofs_ptrs[aggdof]-1
    for l in 1:plength
      coeffsl = coeffs.data[l]
      for b in 1:size(proj,2)
        coeff = z
        for c in 1:size(coeffsl,2)
          coeff += coeffsl[ldof,c]*proj[c,b]
        end
        aggdof_to_coeffs_data[p+b,l] = coeff
      end
    end
  end

  aggdof_to_coeffs = BidimensionalTable(aggdof_to_coeffs_data,aggdof_to_dofs_ptrs)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs
end
