# iudof_to_idof[id of a unique dof] = list of time ids
function get_iudof_to_idof(
  dofs::AbstractVector,
  times::AbstractVector)

  @assert length(dofs) == length(times) "For this integration domain to work, the
  number of spatial selected by the EIM procedure should be equal to the number of
  temporal entries selected by the EIM procedure"

  dofs_to_count = zeros(Int,maximum(dofs))
  for dof in dofs
    dofs_to_count[dof] += 1
  end

  ptrs = Vector{Int32}(undef,length(dofs)+1)
  for (idof,dof) in enumerate(dofs)
    ptrs[idof+1] = dofs_to_count[dof]
  end
  length_to_ptrs!(ptrs)

  data = Vector{Int32}(undef,ptrs[end]-1)
  for (idof,dof) in enumerate(dofs)
    pini = ptrs[idof]
    count = 0
    for (jdof,_dof) in enumerate(dofs)
      if _dof == dof
        count += 1
        data[pini+count-1] = jdof
      end
    end
  end

  return Table(data,ptrs)
end

function get_max_offset(ptrs::Vector{<:Integer})
  offsets = zeros(Int,length(ptrs)-1)
  for i in eachindex(offsets)
    offsets[i] = ptrs[i+1]-ptrs[i]
  end
  return maximum(offsets)
end

function get_max_offset(a::Table)
  get_max_offset(a.ptrs)
end

function get_cells_to_spacetime_idofs(
  cell_dof_ids::AbstractArray{<:AbstractArray},
  cells::AbstractVector,
  dofs::AbstractVector,
  times::AbstractVector)

  _correct_idof(is,li) = li
  _correct_idof(is::OIdsToIds,li) = is.terms[li]

  cache = array_cache(cell_dof_ids)

  ncells = length(cells)
  ptrs = Vector{Int32}(undef,ncells+1)
  @inbounds for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    ptrs[icell+1] = length(celldofs)
  end
  length_to_ptrs!(ptrs)

  # count number of occurrences
  iudof_to_idof = get_iudof_to_idof(dofs,times)
  ucache = array_cache(iudof_to_idof)
  N = get_max_offset(iudof_to_idof)

  z = zeros(Int32,N)
  data = map(_ -> copy(z),1:ptrs[end]-1)
  for (icell,cell) in enumerate(cells)
    celldofs = getindex!(cache,cell_dof_ids,cell)
    for iudof in eachindex(iudof_to_idof)
      idofs = getindex!(ucache,iudof_to_idof,iudof)
      for (iuidof,idof) in enumerate(idofs)
        dof = dofs[idof]
        for (_icelldof,celldof) in enumerate(celldofs)
          if dof == celldof
            icelldof = _correct_idof(celldofs,_icelldof)
            data[ptrs[icell]-1+icelldof][iuidof] = idof
          end
        end
      end
    end
  end

  Table(map(VectorValue,data),ptrs)
end

function reduced_spacetime_idofs(
  f::FESpace,
  trian::Triangulation,
  cells::AbstractVector,
  dofs::AbstractVector,
  indices_times::AbstractVector)

  cell_dof_ids = get_cell_dof_ids(f,trian)
  idofs = get_cells_to_spacetime_idofs(cell_dof_ids,cells,dofs,indices_times)
  return idofs
end

"""
    struct TransientIntegrationDomain <: IntegrationDomain
      domain_space::IntegrationDomain
      indices_time::Vector{Int32}
    end

Integration domain for a projection operator in a transient problem
"""
struct TransientIntegrationDomain <: IntegrationDomain
  domain_space::IntegrationDomain
  indices_time::Vector{Int32}
end

function RBSteady.vector_domain(
  ::TransientReduction,
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector)

  domain_space = vector_domain(trian,test,rows)
  TransientIntegrationDomain(domain_space,indices_time)
end

function RBSteady.matrix_domain(
  ::TransientReduction,
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector)

  domain_space = matrix_domain(trian,trial,test,rows,cols)
  TransientIntegrationDomain(domain_space,indices_time)
end

function RBSteady.vector_domain(
  ::TTSVDReduction,
  trian::Triangulation,
  test::FESpace,
  rows::AbstractVector,
  indices_time::AbstractVector)

  cells = reduced_cells(test,trian,rows)
  irows = reduced_spacetime_idofs(test,trian,cells,rows,indices_time)
  domain_space = VectorDomain(cells,irows)
  TransientIntegrationDomain(domain_space,indices_time)
end

function RBSteady.matrix_domain(
  ::TTSVDReduction,
  trian::Triangulation,
  trial::FESpace,
  test::FESpace,
  rows::AbstractVector,
  cols::AbstractVector,
  indices_time::AbstractVector)

  cells_trial = reduced_cells(trial,trian,cols)
  cells_test = reduced_cells(test,trian,rows)
  cells = union(cells_trial,cells_test)
  icols = reduced_spacetime_idofs(trial,trian,cells,cols,indices_time)
  irows = reduced_spacetime_idofs(test,trian,cells,rows,indices_time)
  domain_space = MatrixDomain(cells,irows,icols)
  TransientIntegrationDomain(domain_space,indices_time)
end

RBSteady.get_integration_cells(i::TransientIntegrationDomain) = get_integration_cells(i.domain_space)
RBSteady.get_cellids_rows(i::TransientIntegrationDomain) = get_cellids_rows(i.domain_space)
RBSteady.get_cellids_cols(i::TransientIntegrationDomain) = get_cellids_cols(i.domain_space)
get_integration_domain_space(i::TransientIntegrationDomain) = i.domain_space
get_indices_time(i::TransientIntegrationDomain) = i.indices_time

function get_itimes(i::TransientIntegrationDomain,ids::AbstractVector)::Vector{Int}
  idsi = get_indices_time(i)
  filter(!isnothing,indexin(idsi,ids))
end
