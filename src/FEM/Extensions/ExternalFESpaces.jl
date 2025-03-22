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
