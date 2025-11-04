"""
    struct OIdsToIds{T,S} <: AbstractVector{T}
      indices::Vector{T}
      terms::Vector{S}
    end
"""
struct OIdsToIds{T,S} <: AbstractVector{T}
  indices::Vector{T}
  terms::Vector{S}
end

Base.size(a::OIdsToIds) = size(a.indices)
Base.getindex(a::OIdsToIds,i::Integer) = getindex(a.indices,i)
Base.setindex!(a::OIdsToIds,v,i::Integer) = setindex!(a.indices,v,i)

function Base.similar(a::OIdsToIds{T},::Type{T′},s::Dims{1}) where {T,T′}
  indices′ = similar(a.indices,T′,s...)
  OIdsToIds(indices′,a.terms)
end

"""
    struct DofsToODofs{D,P,V} <: Map
      b::LagrangianDofBasis{P,V}
      odof_to_dof::Vector{Int8}
      node_and_comps_to_odof::Array{V,D}
      orders::NTuple{D,Int}
    end

Map used to convert a DOF of a standard `FESpace` in [`Gridap`](@ref) to a DOF
belonging to a space whose DOFs are lexicographically-ordered
"""
struct DofsToODofs{D,P,V} <: Map
  b::LagrangianDofBasis{P,V}
  odof_to_dof::Vector{Int8}
  node_and_comps_to_odof::Array{V,D}
  orders::NTuple{D,Int}
end

function DofsToODofs(
  b::LagrangianDofBasis,
  node_and_comps_to_odof::AbstractArray,
  orders::Tuple)

  odof_to_dof = _local_odof_to_dof(b,orders)
  DofsToODofs(b,odof_to_dof,node_and_comps_to_odof,orders)
end

function DofsToODofs(fe_dof_basis::AbstractVector{<:LagrangianDofBasis},args...)
  DofsToODofs(testitem(fe_dof_basis),args...)
end

function DofsToODofs(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function get_ndofs(k::DofsToODofs{D,P}) where {D,P}
  ncomps = num_components(P)
  nnodes = length(k.node_and_comps_to_odof)
  ncomps*nnodes
end

function get_odof(k::DofsToODofs{D,P},dof::Integer) where {D,P}
  nnodes = length(k.node_and_comps_to_odof)
  comp = slow_index(dof,nnodes)
  node = fast_index(dof,nnodes)
  k.node_and_comps_to_odof[node][comp]
end

function Arrays.return_cache(k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  local_ndofs = length(k.odof_to_dof)
  zeros(Int32,local_ndofs)
end

function Arrays.evaluate!(cache,k::DofsToODofs{D},cell::CartesianIndex{D}) where D
  first_new_node = k.orders .* (Tuple(cell) .- 1) .+ 1
  onodes_range = map(enumerate(first_new_node)) do (i,ni)
    ni:ni+k.orders[i]
  end
  local_comps_to_odofs = view(k.node_and_comps_to_odof,onodes_range...)
  local_nnodes = length(k.b.node_and_comp_to_dof)
  for (node,comps_to_odof) in enumerate(local_comps_to_odofs)
    for comp in k.b.dof_to_comp
      odof = comps_to_odof[comp]
      cache[node+(comp-1)*local_nnodes] = odof
    end
  end
  return cache
end

function Arrays.return_value(k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  evaluate(k,dofs,o)
end

function Arrays.return_cache(k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  c = return_cache(k,dofs.indices,o)
  odofs = OIdsToIds(evaluate(k,dofs.indices,o),dofs.terms)
  c,odofs
end

function Arrays.evaluate!(cache,k::Broadcasting{typeof(_sum_if_first_positive)},dofs::OIdsToIds,o::Integer)
  c,odofs = cache
  r = evaluate!(c,k,dofs.indices,o)
  copyto!(odofs.indices,r)
  odofs
end

"""
    struct OReindex <: Map end
"""
struct OReindex <: Map end

function Arrays.return_value(k::OReindex,indices,values)
  values
end

function Arrays.return_cache(k::OReindex,indices::AbstractVector,values::AbstractVector)
  @check length(values) == length(indices)
  similar(values)
end

function Arrays.evaluate!(cache,k::OReindex,indices::AbstractVector,values::AbstractVector)
  for (i,oi) in enumerate(indices)
    cache[oi] = values[i]
  end
  return cache
end

struct OTable{T,S,A<:Table{T},B<:Table{S}} <: AbstractVector{OIdsToIds{T,S}}
  values::A
  terms::B
end

function OTable(vals::AbstractVector{<:AbstractVector},odof_to_dof::AbstractVector)
  values = Table(vals)
  data = repeat(odof_to_dof,length(values))
  ptrs = copy(values.ptrs)
  terms = Table(data,ptrs)
  OTable(values,terms)
end

function OTable(cell_odofs::LazyArray{<:Fill{<:DofsToODofs}})
  values = Table(cell_odofs)
  k = first(cell_odofs.maps)
  OTable(values,k.odof_to_dof)
end

Base.size(a::OTable) = size(a.values)
Base.IndexStyle(::Type{<:OTable}) = IndexLinear()

function Base.getproperty(a::OTable,sym::Symbol)
  if sym in (:data,:ptrs)
    getproperty(a.values,sym)
  else
    getfield(a,sym)
  end
end

Base.view(a::OTable,i::Integer) = OIdsToIds(view(a.values,i),view(a.terms,i))
Base.view(a::OTable,ids::UnitRange{<:Integer}) = OTable(view(a.values,ids),view(a.terms,ids))
Base.getindex(a::OTable,i::Integer) = OIdsToIds(getindex(a.values,i),getindex(a.terms,i))
Base.getindex(a::OTable,ids::UnitRange{<:Integer}) = OTable(getindex(a.values,ids),getindex(a.terms,ids))

function Arrays.array_cache(a::OTable)
  valscache = array_cache(a.values)
  termcache = array_cache(a.terms)
  return valscache,termcache
end

function Arrays.getindex!(c,a::OTable,i::Integer)
  valscache,termcache = c
  pini = a.values.ptrs[i]
  l = a.values.ptrs[i+1] - pini
  setsize!(valscache,(l,))
  setsize!(termcache,(l,))
  pini -= 1
  v = valscache.array
  t = termcache.array
  for j in 1:l
    @inbounds v[j] = a.values.data[pini+j]
    @inbounds t[j] = a.terms.data[pini+j]
  end
  OIdsToIds(v,t)
end

inverse_table(a::OTable) = inverse_table(a.values)

function get_local_ordering(f::SingleFieldFESpace)
  get_local_ordering(get_cell_dof_ids(f))
end

function get_local_ordering(a::AbstractArray)
  get_local_ordering(Table(a))
end

function get_local_ordering(a::Table)
  terms = copy(a)
  for cell in 1:length(terms)
    pini = terms.ptrs[cell]
    pend = terms.ptrs[cell+1]-1
    for (ldof,p) in enumerate(pini:pend)
      terms.data[p] = ldof
    end
  end
  return terms
end

function get_local_ordering(a::OTable)
  a.terms
end

function get_bg_dof_to_dof!(
  bg_fdof_to_fdof,bg_ddof_to_ddof,
  bg_cell_ids::OTable,
  cell_ids::AbstractArray,
  cell_to_bg_cell::AbstractVector
  )

  oldof_to_ldof = get_local_ordering(bg_cell_ids)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  ocache = array_cache(oldof_to_ldof)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_odofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    ldofs = getindex!(ocache,oldof_to_ldof,cell)
    lodofs = invperm(ldofs)
    for (oldof,dof) in enumerate(dofs)
      bg_dof = bg_odofs[lodofs[oldof]]
      if bg_dof > 0
        @check dof > 0
        bg_fdof_to_fdof[bg_dof] = dof
      else
        @check dof < 0
        bg_ddof_to_ddof[-bg_dof] = dof
      end
    end
  end
  return bg_fdof_to_fdof,bg_ddof_to_ddof
end

function get_dof_to_bg_dof!(
  fdof_to_bg_fdof,ddof_to_bg_ddof,
  bg_cell_ids::OTable,
  cell_ids::AbstractArray,
  cell_to_bg_cell::AbstractVector)

  oldof_to_ldof = get_local_ordering(bg_cell_ids)
  bg_cache = array_cache(bg_cell_ids)
  cache = array_cache(cell_ids)
  ocache = array_cache(oldof_to_ldof)
  for (cell,bg_cell) in enumerate(cell_to_bg_cell)
    bg_odofs = getindex!(bg_cache,bg_cell_ids,bg_cell)
    dofs = getindex!(cache,cell_ids,cell)
    ldofs = getindex!(ocache,oldof_to_ldof,cell)
    lodofs = invperm(ldofs)
    for (oldof,dof) in enumerate(dofs)
      bg_dof = bg_odofs[lodofs[oldof]]
      if dof > 0
        @check bg_dof > 0
        fdof_to_bg_fdof[dof] = bg_dof
      else
        @check bg_dof < 0
        ddof_to_bg_ddof[-dof] = bg_dof
      end
    end
  end
  return fdof_to_bg_fdof,ddof_to_bg_ddof
end

# Assembly-related functions

@inline function Algebra.add_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  add_ordered_entries!(combine,A,vs,is,js)
end

for T in (:Any,:(Algebra.ArrayCounter))
  @eval begin
    @inline function Algebra.add_entries!(combine::Function,A::$T,vs,is::OIdsToIds)
      add_ordered_entries!(combine,A,vs,is)
    end
  end
end

"""
    add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)

Adds several ordered entries only for positive input indices. Returns `A`
"""
@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds,js::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices,js.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds,js::OIdsToIds)
  for (lj,j) in enumerate(js)
    if j>0
      ljp = js.terms[lj]
      for (li,i) in enumerate(is)
        if i>0
          lip = is.terms[li]
          vij = vs[lip,ljp]
          add_entry!(combine,A,vij,i,j)
        end
      end
    end
  end
  A
end

@inline function add_ordered_entries!(combine::Function,A,vs::Nothing,is::OIdsToIds)
  Algebra._add_entries!(combine,A,vs,is.indices)
end

@inline function add_ordered_entries!(combine::Function,A,vs,is::OIdsToIds)
  for (li,i) in enumerate(is)
    if i>0
      lip = is.terms[li]
      vi = vs[lip]
      add_entry!(A,vi,i)
    end
  end
  A
end

# utils

function _local_odof_to_dof(fe_dof_basis::AbstractVector{<:Dof},args...)
  @notimplemented "This function is only implemented for Lagrangian dof bases"
end

function _local_odof_to_dof(fe_dof_basis::Fill{<:LagrangianDofBasis},orders::NTuple{D,Int}) where D
  _local_odof_to_dof(testitem(fe_dof_basis),orders)
end

function _local_odof_to_dof(b::LagrangianDofBasis,orders::NTuple{D,Int}) where D
  nnodes = length(b.node_and_comp_to_dof)
  ndofs = length(b.dof_to_comp)

  p = cubic_polytope(Val(D))
  _nodes, = Gridap.ReferenceFEs._compute_nodes(p,orders)
  node_to_pnode = Gridap.ReferenceFEs._coords_to_terms(_nodes,orders)
  node_to_pnode_linear = LinearIndices(orders.+1)[node_to_pnode]

  odof_to_dof = zeros(Int8,ndofs)
  for (inode,ipnode) in enumerate(node_to_pnode_linear)
    for icomp in b.dof_to_comp
      local_shift = (icomp-1)*nnodes
      odof_to_dof[local_shift+ipnode] = local_shift + inode
    end
  end

  return odof_to_dof
end
