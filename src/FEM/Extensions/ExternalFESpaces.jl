function ExternalFESpace(
  bg_space::SingleFieldFESpace,
  int_act_space::SingleFieldFESpace,
  ext_act_space::SingleFieldFESpace)

  return ext_act_space
end

function ExternalFESpace(
  bg_space::SingleFieldFESpace,
  int_agg_space::FESpaceWithLinearConstraints,
  ext_act_space::SingleFieldFESpace,
  bg_cell_to_ext_bg_cell::AbstractVector
  )

  in_dof_to_bg_dof = get_fdof_to_bg_fdof(bg_space,int_agg_space)
  cutout_dof_to_bg_dof = get_fdof_to_bg_fdof(bg_space,ext_act_space)
  aggout_dof_to_bg_dof = intersect(in_dof_to_bg_dof,cutout_dof_to_bg_dof)
  dof_to_aggout_dof = get_bg_fdof_to_fdof(bg_space,ext_act_space,aggout_dof_to_bg_dof)

  shfns_g = get_fe_basis(ext_act_space)
  dofs_g = get_fe_dof_basis(ext_act_space)
  bg_cell_to_gcell = 1:length(bg_cell_to_ext_bg_cell)

  ExternalAgFEMSpace(
    ext_act_space,
    bg_cell_to_ext_bg_cell,
    dof_to_aggout_dof,
    shfns_g,
    dofs_g)
end

function ExternalAgFEMSpace(
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  dof_to_adof::AbstractVector,
  shfns_g::CellField,
  dofs_g::CellDof,
  bgcell_to_gcell::AbstractVector=1:length(bgcell_to_bgcellin)
  )

  # Triangulation made of active cells
  trian_a = get_triangulation(f)

  # Build root cell map (i.e. aggregates) in terms of active cell ids
  D = num_cell_dims(trian_a)
  glue = get_glue(trian_a,Val(D))
  acell_to_bgcell = glue.tface_to_mface
  bgcell_to_acell = glue.mface_to_tface
  acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
  acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
  acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

  # Build shape funs of g by replacing local funs in cut cells by the ones at the root
  # This needs to be done with shape functions in the physical domain
  # otherwise shape funs in cut and root cells are the same
  acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
  acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
  root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

  # Compute data needed to compute the constraints
  dofs_f = get_fe_dof_basis(f)
  shfns_f = get_fe_basis(f)
  acell_to_coeffs = dofs_f(root_shfns_g)
  acell_to_proj = dofs_g(shfns_f)
  acell_to_dof_ids = get_cell_dof_ids(f)
  dof_ids_to_acell = inverse_table(acell_to_dof_ids)
  dof_to_acell_to_ldof_ids = get_ldof_ids_to_acell(acell_to_dof_ids,dof_ids_to_acell)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs = _setup_extagfem_constraints(
    num_free_dofs(f),
    acell_to_acellin,
    acell_to_dof_ids,
    acell_to_coeffs,
    acell_to_proj,
    acell_to_gcell,
    dof_to_adof,
    dof_ids_to_acell,
    dof_to_acell_to_ldof_ids)

  FESpaceWithLinearConstraints(aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,f)
end

function _setup_extagfem_constraints(
  n_fdofs,
  acell_to_acellin,
  acell_to_dof_ids,
  acell_to_coeffs,
  acell_to_proj,
  acell_to_gcell,
  aggdof_to_fdof,
  dof_ids_to_acell,
  dof_to_acell_to_ldof_ids
  )

  n_acells = length(acell_to_acellin)
  fdof_to_acell = zeros(Int32,n_fdofs)
  fdof_to_ldof = zeros(Int16,n_fdofs)
  cache = array_cache(acell_to_dof_ids)
  dcache = array_cache(dof_ids_to_acell)
  lcache = array_cache(dof_to_acell_to_ldof_ids)
  for fdof in aggdof_to_fdof
    acells = getindex!(dcache,dof_ids_to_acell,fdof)
    acell_to_ldofs = getindex!(lcache,dof_to_acell_to_ldof_ids,fdof)
    for (icell,acell) in enumerate(acells)
      ldof = acell_to_ldofs[icell]
      acellin = acell_to_acellin[acell]
      @assert acell != acellin
      gcell = acell_to_gcell[acell]
      acell_dof = fdof_to_acell[fdof]
      if acell_dof == 0 || gcell > acell_to_gcell[acell_dof]
        fdof_to_acell[fdof] = acell
        fdof_to_ldof[fdof] = ldof
      end
    end
  end

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

  T = eltype(eltype(acell_to_coeffs))
  z = zero(T)

  aggdof_to_coefs_data = zeros(T,ndata)
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
      aggdof_to_coefs_data[p+b] = coeff
    end
  end

  aggdof_to_coeffs = Table(aggdof_to_coefs_data,aggdof_to_dofs_ptrs)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs
end

function get_ldof_ids_to_acell(cell_2_dof::Table,dof_2_cell::Table)
  ldof_dof_2_cell = copy(dof_2_cell)
  for dof in 1:length(dof_2_cell)
    pini = dof_2_cell.ptrs[dof]
    pend = dof_2_cell.ptrs[dof+1]-1
    for p in pini:pend
      cell = dof_2_cell.data[p]
      qini = cell_2_dof.ptrs[cell]
      qend = cell_2_dof.ptrs[cell+1]-1
      for (ldof,q) in enumerate(qini:qend)
        if cell_2_dof.data[q] == dof
          ldof_dof_2_cell.data[p] = ldof
          break
        end
      end
    end
  end
  return ldof_dof_2_cell
end


function OrderedAgFEMSpace(
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  g::SingleFieldFESpace=f,
  args...
  )

  @assert get_triangulation(f) === get_triangulation(g)
  OrderedAgFEMSpace(f,bgcell_to_bgcellin,get_fe_basis(g),get_fe_dof_basis(g),args...)
end

# Note: cell is in fact bgcell in this function since f will usually be an ExtendedFESpace
function OrderedAgFEMSpace(
  f::SingleFieldFESpace,
  bgcell_to_bgcellin::AbstractVector,
  shfns_g::CellField,
  dofs_g::CellDof,
  bgcell_to_gcell::AbstractVector=1:length(bgcell_to_bgcellin)
  )

  # Triangulation made of active cells
  trian_a = get_triangulation(f)

  # Build root cell map (i.e. aggregates) in terms of active cell ids
  D = num_cell_dims(trian_a)
  glue = get_glue(trian_a,Val(D))
  acell_to_bgcell = glue.tface_to_mface
  bgcell_to_acell = glue.mface_to_tface
  acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
  acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
  acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

  # Build shape funs of g by replacing local funs in cut cells by the ones at the root
  # This needs to be done with shape functions in the physical domain
  # otherwise shape funs in cut and root cells are the same
  acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
  acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
  root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

  # Compute data needed to compute the constraints
  dofs_f = get_fe_dof_basis(f)
  shfns_f = get_fe_basis(f)
  acell_to_coeffs = dofs_f(root_shfns_g)
  acell_to_proj = dofs_g(shfns_f)
  acell_to_dof_ids = get_cell_dof_ids(f)

  aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs = AgFEM._setup_agfem_constraints(
    num_free_dofs(f),
    acell_to_acellin,
    acell_to_dof_ids,
    acell_to_coeffs,
    acell_to_proj,
    acell_to_gcell)

  OrderedFESpaceWithLinearConstraints(
    aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,f,acell_to_acellin)
end

struct OrderedFESpaceWithLinearConstraints{S<:SingleFieldFESpace} <: SingleFieldFESpace
  space::S
  n_fdofs::Int
  n_fmdofs::Int
  mDOF_to_DOF::Vector
  DOF_to_mDOFs::Table
  DOF_to_coeffs::Table
  cell_to_incell::Vector
  incell_to_lmdof_to_mdof::Table
  cutcell_to_lmdof_to_mdof::Table
  cell_to_ldof_to_dof::Table
end

function OrderedFESpaceWithLinearConstraints(
  sDOF_to_dof::AbstractVector{<:Integer},
  sDOF_to_dofs::Table,
  sDOF_to_coeffs::Table,
  space::SingleFieldFESpace,
  acell_to_acellin
  )

  n_fdofs = num_free_dofs(space)
  n_ddofs = num_dirichlet_dofs(space)
  n_DOFs = n_fdofs + n_ddofs

  DOF_to_DOFs,DOF_to_coeffs = FESpaces._prepare_DOF_to_DOFs(
    sDOF_to_dof,sDOF_to_dofs,sDOF_to_coeffs,n_fdofs,n_DOFs)

  OrderedFESpaceWithLinearConstraints!(DOF_to_DOFs,DOF_to_coeffs,space,acell_to_acellin)
end

function OrderedFESpaceWithLinearConstraints!(
  DOF_to_DOFs::Table,
  DOF_to_coeffs::Table,
  space::SingleFieldFESpace,
  acell_to_acellin
  )

  n_fdofs = num_free_dofs(space)
  mDOF_to_DOF,n_fmdofs = FESpaces._find_master_dofs(DOF_to_DOFs,n_fdofs)
  DOF_to_mDOFs = FESpaces._renumber_constraints!(DOF_to_DOFs,mDOF_to_DOF)
  cell_to_ldof_to_dof = Table(get_cell_dof_ids(space))
  incell_to_lmdof_to_mdof,cutcell_to_lmdof_to_mdof = _setup_ordered_cell_to_lmdof_to_mdof(
    cell_to_ldof_to_dof,
    acell_to_acellin,
    DOF_to_mDOFs,
    n_fdofs,
    n_fmdofs)

  OrderedFESpaceWithLinearConstraints(
    space,
    n_fdofs,
    n_fmdofs,
    mDOF_to_DOF,
    DOF_to_mDOFs,
    DOF_to_coeffs,
    acell_to_acellin,
    incell_to_lmdof_to_mdof,
    cutcell_to_lmdof_to_mdof,
    cell_to_ldof_to_dof)

end

function _setup_ordered_cell_to_lmdof_to_mdof(args...)
  incell_to_lmdof_to_mdof = _get_incell_to_lmdof_to_mdof(args...)
  cutcell_to_lmdof_to_mdof = _get_cutcell_to_lmdof_to_mdof(args...)
  return (incell_to_lmdof_to_mdof,cutcell_to_lmdof_to_mdof)
end

function _get_incell_to_lmdof_to_mdof(
  cell_to_ldof_to_dof,
  acell_to_acellin,
  DOF_to_mDOFs,
  n_fdofs,
  n_fmdofs
  )

  acellin = unique(acell_to_acellin)
  n_incells = length(acell_to_acellin)
  cell_to_lmdof_to_mdof_ptrs = zeros(eltype(cell_to_ldof_to_dof.ptrs),n_incells+1)

  for (icell,cell) in enumerate(acell_to_acellin)
    pini = cell_to_ldof_to_dof.ptrs[cell]
    pend = cell_to_ldof_to_dof.ptrs[cell+1]
    cell_to_lmdof_to_mdof_ptrs[icell+1] = pend-pini
  end

  length_to_ptrs!(cell_to_lmdof_to_mdof_ptrs)
  ndata = cell_to_lmdof_to_mdof_ptrs[end]-1
  cell_to_lmdof_to_mdof_data = zeros(eltype(cell_to_ldof_to_dof.data),ndata)

  for (icell,cell) in enumerate(acellin)
    pini = cell_to_ldof_to_dof.ptrs[cell]
    pend = cell_to_ldof_to_dof.ptrs[cell+1]-1
    o = cell_to_lmdof_to_mdof_ptrs[icell]-1
    for (lmdof,p) in enumerate(pini:pend)
      dof = cell_to_ldof_to_dof.data[p]
      DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
      qini = DOF_to_mDOFs.ptrs[DOF]
      qend = DOF_to_mDOFs.ptrs[DOF+1]-1
      @assert qend == qini
      mDOF = DOF_to_mDOFs.data[qini]
      mdof = FESpaces._DOF_to_dof(mDOF,n_fmdofs)
      cell_to_lmdof_to_mdof_data[o+lmdof] = mdof
    end
  end

  Table(cell_to_lmdof_to_mdof_data,cell_to_lmdof_to_mdof_ptrs)
end

function _get_cutcell_to_lmdof_to_mdof(
  cell_to_ldof_to_dof,
  acell_to_acellin,
  DOF_to_mDOFs,
  n_fdofs,
  n_fmdofs
  )

  n_acells = length(cell_to_ldof_to_dof)
  acellcut = setdiff(1:n_acells,acell_to_acellin)
  n_acellcut = length(acellcut)

  cell_to_lmdof_to_mdof_ptrs = zeros(eltype(cell_to_ldof_to_dof.ptrs),n_acellcut+1)

  for (icell,cell) in enumerate(acellcut)
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
    cell_to_lmdof_to_mdof_ptrs[icell+1] = length(mdofs)
  end

  length_to_ptrs!(cell_to_lmdof_to_mdof_ptrs)
  ndata = cell_to_lmdof_to_mdof_ptrs[end]-1
  cell_to_lmdof_to_mdof_data = zeros(eltype(cell_to_ldof_to_dof.data),ndata)

  for (icell,cell) in enumerate(acellcut)
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
    o = cell_to_lmdof_to_mdof_ptrs[icell]-1
    for (lmdof,mdof) in enumerate(mdofs)
      cell_to_lmdof_to_mdof_data[o+lmdof] = mdof
    end
  end

  Table(cell_to_lmdof_to_mdof_data,cell_to_lmdof_to_mdof_ptrs)
end

function FESpaces.get_cell_dof_ids(f::OrderedFESpaceWithLinearConstraints)
  f.incell_to_lmdof_to_mdof
end

function FESpaces.get_fe_dof_basis(f::OrderedFESpaceWithLinearConstraints)
  get_fe_dof_basis(f.space)
end

function FESpaces.get_dirichlet_dof_ids(f::OrderedFESpaceWithLinearConstraints)
  Base.OneTo(length(f.mDOF_to_DOF) - f.n_fmdofs)
end

function FESpaces.num_dirichlet_tags(f::OrderedFESpaceWithLinearConstraints)
  num_dirichlet_tags(f.space)
end

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

function FESpaces.scatter_free_and_dirichlet_values(
  f::OrderedFESpaceWithLinearConstraints,
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

function FESpaces.gather_free_and_dirichlet_values(
  f::OrderedFESpaceWithLinearConstraints,
  cell_to_ludof_to_val
  )

  fdof_to_val,ddof_to_val = gather_free_and_dirichlet_values(f.space,cell_to_ludof_to_val)
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
  f::OrderedFESpaceWithLinearConstraints,
  cell_to_ludof_to_val
  )

  fdof_to_val,ddof_to_val = gather_free_and_dirichlet_values(f.space,cell_to_ludof_to_val)
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

# Implementation of FESpace interface

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
  cell_to_mask, = _get_cell_to_in_cut_info(f)
  return cell_to_mask
end

function FESpaces.get_cell_constraints(f::OrderedFESpaceWithLinearConstraints)
  cell_to_mask,cell_to_cellcut = _get_cell_to_in_cut_info(f)

  k = OrderedLinearConstraintsMap(
    f.cutcell_to_lmdof_to_mdof,
    f.DOF_to_mDOFs,
    f.DOF_to_coeffs,
    length(f.mDOF_to_DOF),
    f.n_fmdofs,
    f.n_fdofs
    )

  cell_to_mat = get_cell_constraints(f.space)
  lazy_map(k,f.incell_to_lmdof_to_mdof,f.cell_to_ldof_to_dof,cell_to_mat,cell_to_mask,cell_to_cellcut)
end

struct OrderedLinearConstraintsMap{A,B,C} <: Map
  f::LinearConstraintsMap{A,B}
  cutcell_to_lmdof_to_mdof::C
end

function OrderedLinearConstraintsMap(cutcell_to_lmdof_to_mdof,args...)
  f = LinearConstraintsMap(args...)
  OrderedLinearConstraintsMap(f,cutcell_to_lmdof_to_mdof)
end

function Arrays.return_cache(
  k::OrderedLinearConstraintsMap,lmdof_to_mdof,ldof_to_dof,mat,mask,cellcut
  )

  return_cache(k.f,lmdof_to_mdof,ldof_to_dof,mat)
end

function Arrays.evaluate!(
  cache,k::OrderedLinearConstraintsMap,lmdof_to_mdof,ldof_to_dof,mat,mask,cellcut
  )

  if mask
    lmdof_to_mdof′ = k.cutcell_to_lmdof_to_mdof[cellcut]
    evaluate!(cache,k.f,lmdof_to_mdof′,ldof_to_dof,mat)
  else
    m1,m2,mDOF_to_lmdof = cache
    n_lmdofs = length(lmdof_to_mdof)
    n_ludofs = size(mat,2)

    setsize!(m2,(n_lmdofs,n_ludofs))
    a2 = m2.array
    copyto!(a2,mat)
    a2
  end
end

# struct MoveConstrainedVecVals{A,B,C,D} <: Map
#   ccell_to_cdofs::A
#   cell_to_dofs::B
#   dof_to_cells::C
#   vals::D
# end

# function Arrays.return_cache(k::MoveConstrainedVecVals,cell,ccell)
#   @assert iszero(cell) != iszero(ccell)
#   c1 = array_cache(k.vals)
#   c2 = array_cache(k.vals)
#   c3 = array_cache(k.vals)
#   Tvals = eltype(eltype(k.vals))
#   nvals = length(testitem(k.cell_to_dofs))
#   move_vals = zeros(Tvals,nvals)
#   c1,c2,move_vals
# end

# function Arrays.evaluate!(cache,k::MoveConstrainedVecVals,cell,ccell)
#   c1,c2,move_vals = cache
#   if iszero(cell)
#     cdofs = k.ccell_to_cdofs[ccell]
#     cvals = getindex!(c1,k.vals,ccell)
#     for (ci,cdof) in enumerate(cdofs)
#       cvi = cvals[ci]
#       cells = k.dof_to_cells[cdof]
#       _cell = first(cells) # it does not matter where we move the constrained values
#       dofs = k.cell_to_dofs[_cell]
#       vals = getindex!(c2,k.vals,_cell)
#       for (i,dof) in enumerate(dofs)
#         if dof == cdof
#           move_vals[i] = vals[i] + cvi
#           break
#         end
#       end
#     end
#   else
#     move_vals = getindex!(c1,k.vals,cell)
#   end
#   return move_vals
# end

# utils

struct OrderedAppendTables{T,A<:Table{T},B<:Table{T}} <: AbstractVector{Vector{T}}
  a::A
  b::B
  id_to_ida::Vector{Int32}
  id_to_idb::Vector{Int32}
end

function _get_cell_to_in_cut_info(f::OrderedFESpaceWithLinearConstraints)
  IN = false
  CUT = true
  cell_to_incell = f.cell_to_incell
  ncells = length(cell_to_incell)
  incells = unique(cell_to_incell)
  cutcells = setdiff(1:ncells,incells)
  cell_to_mask = fill(IN,ncells)
  cell_to_cellcut = zeros(Int32,ncells)
  for (icut,cell) in enumerate(cutcells)
    cell_to_mask[cell] = CUT
    cell_to_cellcut[cell] = icut
  end
  return cell_to_mask,cell_to_cellcut
end
