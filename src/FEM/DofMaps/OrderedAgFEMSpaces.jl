function GridapEmbedded.AgFEMSpace(
  bg_f::OrderedFESpace,
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  g::SingleFieldFESpace=f,
  args...)

  acell_to_terms = get_term_to_bg_terms(bg_f,f)
  OrderedAgFEMSpace(f,bgcell_to_bgcellin,acell_to_terms,g,args...)
end

function OrderedAgFEMSpace(
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  acell_to_terms::Table,
  g::SingleFieldFESpace=f,
  args...)

  @assert get_triangulation(f) === get_triangulation(g)
  OrderedAgFEMSpace(f,bgcell_to_bgcellin,acell_to_terms,get_fe_basis(g),get_fe_dof_basis(g),args...)
end

function OrderedAgFEMSpace(
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  acell_to_terms::Table,
  shfns_g::CellField,
  dofs_g::CellDof,
  bgcell_to_gcell::AbstractVector=1:length(bgcell_to_bgcellin)
  )

  trian_a = get_triangulation(f)

  D = num_cell_dims(trian_a)
  glue = get_glue(trian_a,Val(D))
  acell_to_bgcell = glue.tface_to_mface
  bgcell_to_acell = glue.mface_to_tface
  acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
  acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
  acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

  acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
  acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
  root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

  dofs_f = get_fe_dof_basis(f)
  shfns_f = get_fe_basis(f)
  acell_to_coeffs = dofs_f(root_shfns_g)
  acell_to_proj = dofs_g(shfns_f)
  acell_to_dof_ids = get_cell_dof_ids(f)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,args = _setup_oagfem_constraints(
    num_free_dofs(f),
    acell_to_acellin,
    acell_to_terms,
    acell_to_dof_ids,
    acell_to_coeffs,
    acell_to_proj,
    acell_to_gcell)

  OrderedFESpaceWithLinearConstraints(
    f,
    aggdof_to_fdof,
    aggdof_to_dofs,
    aggdof_to_coeffs,
    acell_to_terms,
    args...)
end

function _setup_oagfem_constraints(
  n_fdofs,
  acell_to_acellin,
  acell_to_terms,
  acell_to_dof_ids,
  acell_to_coeffs,
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
    ldof = fdof_to_ldof[fdof]
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

  T = eltype(eltype(acell_to_coeffs))
  z = zero(T)

  aggdof_to_coeffs_data = zeros(T,ndata)
  for aggdof in 1:n_aggdofs
    fdof = aggdof_to_fdof[aggdof]
    acell = fdof_to_acell[fdof]
    coeffs = getindex!(cache2,acell_to_coeffs,acell)
    proj = getindex!(cache3,acell_to_proj,acell)
    ldof = fdof_to_ldof[fdof]
    p = aggdof_to_dofs_ptrs[aggdof]-1
    for b in 1:size(proj,2)
      coeff = z
      for c in 1:size(coeffs,2)
        coeff += coeffs[ldof,c]*proj[c,b]
      end
      aggdof_to_coeffs_data[p+b] = coeff
    end
  end

  aggdof_to_coeffs = Table(aggdof_to_coeffs_data,aggdof_to_dofs_ptrs)

  args = (fdof_to_acell,fdof_to_isagg,fdof_to_ldof,acell_to_acellin)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,args
end

struct OrderedFESpaceWithLinearConstraints{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  n_fdofs::Int
  n_fmdofs::Int
  mDOF_to_DOF::Vector
  DOF_to_mDOFs::Table
  DOF_to_coeffs::Table
  cell_to_lmdof_to_mdof::Table
  cell_to_lmdof_to_term::Table
  cell_to_ldof_to_dof::Table
end

function OrderedFESpaceWithLinearConstraints(
  space::SingleFieldFESpace,
  sDOF_to_dof::AbstractVector{<:Integer},
  sDOF_to_dofs::Table,
  sDOF_to_coeffs::Table,
  acell_to_terms::Table,
  args...)

  n_fdofs = num_free_dofs(space)
  n_ddofs = num_dirichlet_dofs(space)
  n_DOFs = n_fdofs+n_ddofs

  DOF_to_DOFs,DOF_to_coeffs = FESpaces._prepare_DOF_to_DOFs(
    sDOF_to_dof,
    sDOF_to_dofs,
    sDOF_to_coeffs,
    n_fdofs,
    n_DOFs)

  OrderedFESpaceWithLinearConstraints!(space,DOF_to_DOFs,DOF_to_coeffs,acell_to_terms,args...)
end

function _prepare_oDOF_to_oDOFs(
  sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs,sDOF_to_term,n_fdofs,n_DOFs)

  Tp = eltype(sDOF_to_dofs.ptrs)
  Td = eltype(sDOF_to_dofs.data)
  Tc = eltype(sDOF_to_coeffs.data)
  Tt = eltype(sDOF_to_term)

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
  DOF_to_DOFs_data = zeros(Td,ndata)
  DOF_to_coeffs_data = ones(Tc,ndata)
  DOF_to_term = zeros(Tt,ndata)

  for DOF in 1:n_DOFs
    q = DOF_to_DOFs_ptrs[DOF]
    DOF_to_DOFs_data[q] = DOF
  end

  for sDOF in 1:n_sDOFs
    dof = sDOF_to_dof[sDOF]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    DOF_to_term[DOF] = sDOF_to_term[sDOF]
    q = DOF_to_DOFs_ptrs[DOF]-1
    pini = sDOF_to_dofs.ptrs[sDOF]
    pend = sDOF_to_dofs.ptrs[sDOF+1]-1
    for (i,p) in enumerate(pini:pend)
      _dof = sDOF_to_dofs.data[p]
      _DOF = FESpaces._dof_to_DOF(_dof,n_fdofs)
      coeff = sDOF_to_coeffs.data[p]
      DOF_to_DOFs_data[q+i] = _DOF
      DOF_to_coeffs_data[q+i] = coeff
    end
  end

  DOF_to_DOFs = Table(DOF_to_DOFs_data,DOF_to_DOFs_ptrs)
  DOF_to_coeffs = Table(DOF_to_coeffs_data,DOF_to_DOFs_ptrs)

  DOF_to_DOFs,DOF_to_coeffs,DOF_to_term
end

function OrderedFESpaceWithLinearConstraints!(
  space::SingleFieldFESpace,
  DOF_to_DOFs::Table,
  DOF_to_coeffs::Table,
  cell_to_terms::Table,
  args...)

  n_fdofs = num_free_dofs(space)
  mDOF_to_DOF,n_fmdofs = FESpaces._find_master_dofs(DOF_to_DOFs,n_fdofs)
  DOF_to_mDOFs = FESpaces._renumber_constraints!(DOF_to_DOFs,mDOF_to_DOF)
  cell_to_ldof_to_dof = Table(get_cell_dof_ids(space))
  cell_to_lmdof_to_mdof,cell_to_lmdof_to_isagg = _setup_cell_to_lomdof_to_omdof(
    cell_to_ldof_to_dof,
    DOF_to_mDOFs,
    n_fdofs,
    n_fmdofs)
  cell_to_lmdof_to_term = _setup_cell_to_lmdof_to_term(
    cell_to_ldof_to_dof,
    cell_to_lmdof_to_mdof,
    cell_to_lmdof_to_isagg,
    cell_to_terms,
    n_fdofs,n_fmdofs,
    mDOF_to_DOF,
    args...)

  OrderedFESpaceWithLinearConstraints(
    space,
    n_fdofs,
    n_fmdofs,
    mDOF_to_DOF,
    DOF_to_mDOFs,
    DOF_to_coeffs,
    cell_to_lmdof_to_mdof,
    cell_to_lmdof_to_term,
    cell_to_ldof_to_dof)

end

function _setup_cell_to_lomdof_to_omdof(
  cell_to_ldof_to_dof,DOF_to_mDOFs,n_fdofs,n_fmdofs)

  n_cells = length(cell_to_ldof_to_dof)
  cell_to_lmdof_to_mdof_ptrs = zeros(eltype(cell_to_ldof_to_dof.ptrs),n_cells+1)

  for cell in 1:n_cells
    mdofs = Set{Int}()
    pini = cell_to_ldof_to_dof.ptrs[cell]
    pend = cell_to_ldof_to_dof.ptrs[cell+1]-1
    for p in pini:pend
      dof = cell_to_ldof_to_dof.data[p]
      DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
      qini = DOF_to_mDOFs.ptrs[DOF]
      qend = DOF_to_mDOFs.ptrs[DOF+1]-1
      for q in qini:qend
        mDOF = DOF_to_mDOFs.data[q]
        mdof = FESpaces._DOF_to_dof(mDOF,n_fmdofs)
        push!(mdofs,mdof)
      end
    end
    cell_to_lmdof_to_mdof_ptrs[cell+1] = length(mdofs)
  end

  length_to_ptrs!(cell_to_lmdof_to_mdof_ptrs)
  cell_to_lmdof_to_term_ptrs = copy(cell_to_lmdof_to_mdof_ptrs)

  ndata = cell_to_lmdof_to_mdof_ptrs[end]-1
  cell_to_lmdof_to_mdof_data = zeros(eltype(cell_to_ldof_to_dof.data),ndata)
  cell_to_lmdof_to_isagg_data = zeros(Bool,ndata)

  for cell in 1:n_cells
    mdofs = OrderedDict{Int,Int8}()
    pini = cell_to_ldof_to_dof.ptrs[cell]
    pend = cell_to_ldof_to_dof.ptrs[cell+1]-1
    for p in pini:pend
      dof = cell_to_ldof_to_dof.data[p]
      DOF = _dof_to_DOF(dof,n_fdofs)
      qini = DOF_to_mDOFs.ptrs[DOF]
      qend = DOF_to_mDOFs.ptrs[DOF+1]-1
      isagg = qini != qend
      for q in qini:qend
        mDOF = DOF_to_mDOFs.data[q]
        mdof = _DOF_to_dof(mDOF,n_fmdofs)
        mdofs[mdof] = isagg
      end
    end
    o = cell_to_lmdof_to_mdof_ptrs[cell]-1
    for (lmdof,(mdof,isagg)) in enumerate(mdofs)
      cell_to_lmdof_to_mdof_data[o+lmdof] = mdof
      cell_to_lmdof_to_isagg_data[o+lmdof] = isagg
    end
  end

  cell_to_lmdof_to_mdof = Table(cell_to_lmdof_to_mdof_data,cell_to_lmdof_to_mdof_ptrs)
  cell_to_lmdof_to_isagg = Table(cell_to_lmdof_to_isagg_data,cell_to_lmdof_to_mdof_ptrs)
  return cell_to_lmdof_to_mdof,cell_to_lmdof_to_isagg
end

function _setup_cell_to_lmdof_to_term(
  cell_to_ldof_to_dof,
  cell_to_lmdof_to_mdof,
  acell_to_terms,
  n_fdofs,n_fmdofs,
  mDOF_to_DOF,
  fdof_to_acell,
  fdof_to_ldof,
  acell_to_acellin)

  ndata = cell_to_lmdof_to_mdof.ptrs[end]-1
  cell_to_lmdof_to_term_data = zeros(eltype(acell_to_terms.data),ndata)
  dcache = array_cache(cell_to_ldof_to_dof)
  mcache = array_cache(cell_to_lmdof_to_mdof)
  tcache = array_cache(acell_to_terms)
  for acell in 1:length(cell_to_lmdof_to_mdof)
    acellin = acell_to_acellin[acell]
    dofs = getindex!(dcache,cell_to_ldof_to_dof,acellin)
    mdofs = getindex!(mcache,cell_to_lmdof_to_mdof,acell)
    terms = getindex!(tcache,acell_to_terms,acell)
    for (q,mdof) in enumerate(mdofs)
      mDOF = FESpaces._dof_to_DOF(mdof,n_fmdofs)
      DOF = mDOF_to_DOF[mDOF]
      dof = FESpaces._DOF_to_dof(DOF,n_fdofs)
      if dof > 0
        fdof = dof
        isagg = fdof_to_isagg[fdof]
        if isagg
          for (ldof,dof) in enumerate(dofs)
            if dof == fdof
              term = terms[ldof]
              cell_to_lmdof_to_term_data[q] = term
              break
            end
          end
        else
          ldof = fdof_to_ldof[fdof]
          term = terms[ldof]
          cell_to_lmdof_to_term_data[q] = term
        end
      end
    end
  end
  Table(cell_to_lmdof_to_term_data,cell_to_lmdof_to_mdof.ptrs)
end

function FESpaces.get_cell_dof_ids(f::OrderedFESpaceWithLinearConstraints)
  f.cell_to_lmdof_to_mdof
end

function FESpaces.get_fe_dof_basis(f::OrderedFESpaceWithLinearConstraints)
  get_fe_dof_basis(f.space)
end

FESpaces.get_dof_value_type(f::OrderedFESpaceWithLinearConstraints) = get_dof_value_type(f.space)

FESpaces.get_dirichlet_dof_ids(f::OrderedFESpaceWithLinearConstraints) = Base.OneTo(length(f.mDOF_to_DOF) - f.n_fmdofs)

FESpaces.num_dirichlet_tags(f::OrderedFESpaceWithLinearConstraints) = num_dirichlet_tags(f.space)

function FESpaces.get_dirichlet_dof_tag(f::OrderedFESpaceWithLinearConstraints)
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

function FESpaces.get_dirichlet_dof_values(f::OrderedFESpaceWithLinearConstraints)
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

function FESpaces.scatter_free_and_dirichlet_values(f::OrderedFESpaceWithLinearConstraints,fmdof_to_val,dmdof_to_val)
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

function FESpaces.gather_free_and_dirichlet_values(f::OrderedFESpaceWithLinearConstraints,cell_to_ludof_to_val)
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

function FESpaces.gather_free_and_dirichlet_values!(fmdof_to_val,dmdof_to_val,f::OrderedFESpaceWithLinearConstraints,cell_to_ludof_to_val)
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

function FESpaces.get_triangulation(f::OrderedFESpaceWithLinearConstraints)
  get_triangulation(f.space)
end

FESpaces.get_free_dof_ids(f::OrderedFESpaceWithLinearConstraints) = Base.OneTo(f.n_fmdofs)

function FESpaces.get_vector_type(f::OrderedFESpaceWithLinearConstraints)
  get_vector_type(f.space)
end

function FESpaces.get_fe_basis(f::OrderedFESpaceWithLinearConstraints)
  get_fe_basis(f.space)
end

function FESpaces.get_trial_fe_basis(f::OrderedFESpaceWithLinearConstraints)
  get_trial_fe_basis(f.space)
end

function FESpaces.CellField(f::OrderedFESpaceWithLinearConstraints,cellvals)
  CellField(f.space,cellvals)
end

FESpaces.ConstraintStyle(::Type{<:OrderedFESpaceWithLinearConstraints}) = Constrained()

function FESpaces.get_cell_isconstrained(f::OrderedFESpaceWithLinearConstraints)
  n = length(get_cell_dof_ids(f))
  Fill(true,n)
end

function FESpaces.get_cell_constraints(f::OrderedFESpaceWithLinearConstraints)
  k = FESpaces.LinearConstraintsMap(
    f.DOF_to_mDOFs,
    f.DOF_to_coeffs,
    length(f.mDOF_to_DOF),
    f.n_fmdofs,
    f.n_fdofs)

  cell_to_mat = get_cell_constraints(f.space)
  lazy_map(k,f.cell_to_lmdof_to_mdof,f.cell_to_ldof_to_dof,cell_to_mat)
end

function get_bg_dof_to_dof(bg_f::SingleFieldFESpace,agg_f::OrderedFESpaceWithLinearConstraints)
  act_fdof_to_agg_fdof,act_ddof_to_agg_ddof = get_dof_to_mdof(agg_f)
  bg_fdof_to_act_fdof,bg_ddof_to_act_ddof = get_bg_dof_to_dof(bg_f,agg_f.space)
  bg_fdof_to_agg_fdof = compose_index(bg_fdof_to_act_fdof,act_fdof_to_agg_fdof)
  bg_ddof_to_agg_ddof = compose_index(bg_ddof_to_act_ddof,act_ddof_to_agg_ddof)
  return bg_fdof_to_agg_fdof,bg_ddof_to_agg_ddof
end

function get_dof_to_bg_dof(bg_f::SingleFieldFESpace,agg_f::OrderedFESpaceWithLinearConstraints)
  agg_fdof_to_act_fdof,agg_ddof_to_act_ddof = get_mdof_to_dof(agg_f)
  act_fdof_to_bg_fdof,act_ddof_to_bg_ddof = get_dof_to_bg_dof(bg_f,agg_f.space)
  agg_fdof_to_bg_fdof = compose_index(agg_fdof_to_act_fdof,act_fdof_to_bg_fdof)
  agg_ddof_to_bg_ddof = compose_index(agg_ddof_to_act_ddof,act_ddof_to_bg_ddof)
  return agg_fdof_to_bg_fdof,agg_ddof_to_bg_ddof
end

function get_dof_to_mdof(f::OrderedFESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  fdof_to_mfdof = zeros(T,num_free_dofs(f.space))
  ddof_to_mddof = zeros(T,num_dirichlet_dofs(f.space))
  cache = array_cache(f.DOF_to_mDOFs)
  for DOF in 1:length(f.DOF_to_mDOFs)
    mDOFs = getindex!(cache,f.DOF_to_mDOFs,DOF)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    for mDOF in mDOFs
      mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
      if dof > 0
        fdof_to_mfdof[dof] = mdof
      else
        ddof_to_mddof[-dof] = -mdof
      end
    end
  end
  return fdof_to_mfdof,ddof_to_mddof
end

function get_mdof_to_dof(f::OrderedFESpaceWithLinearConstraints)
  T = eltype(f.mDOF_to_DOF)
  mfdof_to_fdof = zeros(T,num_free_dofs(f))
  mddof_to_ddof = zeros(T,num_dirichlet_dofs(f))
  for mDOF in 1:length(f.mDOF_to_DOF)
    DOF = f.mDOF_to_DOF[mDOF]
    mdof = FESpaces._DOF_to_dof(mDOF,f.n_fmdofs)
    dof = FESpaces._DOF_to_dof(DOF,f.n_fdofs)
    if mdof > 0
      mfdof_to_fdof[mdof] = dof
    else
      mddof_to_ddof[-mdof] = -dof
    end
  end
  return mfdof_to_fdof,mddof_to_ddof
end

function get_bg_dof_to_active_dof(bg_f::SingleFieldFESpace,f::OrderedFESpaceWithLinearConstraints)
  get_bg_dof_to_dof(bg_f,f.space)
end

function get_active_dof_to_bg_dof(bg_f::SingleFieldFESpace,f::OrderedFESpaceWithLinearConstraints)
  get_dof_to_bg_dof(bg_f,f.space)
end

function get_term_to_bg_terms(bg_f::OrderedFESpace,f::OrderedFESpaceWithLinearConstraints)
  f.cell_to_lmdof_to_term
end
