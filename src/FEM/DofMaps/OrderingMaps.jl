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
      odof_to_dof::Vector{Int32}
      node_and_comps_to_odof::Array{V,D}
      orders::NTuple{D,Int}
    end

Map used to convert a DOF of a standard `FESpace` in [`Gridap`](@ref) to a DOF
belonging to a space whose DOFs are lexicographically-ordered
"""
struct DofsToODofs{D,P,V} <: Map
  b::LagrangianDofBasis{P,V}
  odof_to_dof::Vector{Int32}
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

"""
    struct OReindex{T<:Integer} <: Map
      indices::Vector{T}
    end

Map used to reindex according to the vector of integers `indices`
"""
struct OReindex{T<:Integer} <: Map
  indices::Vector{T}
end

function Arrays.return_value(k::OReindex,values)
  values
end

function Arrays.return_cache(k::OReindex,values::AbstractVector)
  @check length(values) == length(k.indices)
  similar(values)
end

function Arrays.evaluate!(cache,k::OReindex,values::AbstractVector)
  for (i,oi) in enumerate(k.indices)
    cache[oi] = values[i]
  end
  return cache
end

struct OTable{T,Vd<:AbstractVector{T},Vp<:AbstractVector} <: AbstractVector{Vector{T}}
  values::Table{T,Vd,Vp}
  terms::Vector{Int32}
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

Base.view(a::OTable,i::Integer) = view(a.values,i)
Base.view(a::OTable,ids::UnitRange{<:Integer}) = OTable(view(a.values,ids),a.terms)
Base.getindex(a::OTable,i) = getindex(a.values,i)

Arrays.array_cache(a::OTable) = array_cache(a.values)
Arrays.getindex!(c,a::OTable,i::Integer) = getindex!(c,a.values,i)

function FESpaces.get_cell_fe_data(fun,sface_to_data::OTable,sglue::FaceToFaceGlue,tglue::FaceToFaceGlue)
  error("need to implement this")
  # mface_to_sface = sglue.mface_to_tface
  # tface_to_mface = tglue.tface_to_mface
  # mface_to_data = extend(sface_to_data,mface_to_sface)
  # tface_to_data = lazy_map(Reindex(mface_to_data),tface_to_mface)
  # tface_to_data
end

# Assembly-related functions

@noinline function FESpaces._numeric_loop_matrix!(mat,caches,cell_vals,cell_rows::OTable,cell_cols::OTable)
  add_cache,vals_cache,rows_cache,cols_cache = caches
  row_terms,col_terms = cell_rows.terms,cell_cols.terms
  add! = Arrays.AddEntriesMap(+)
  for cell in 1:length(cell_cols)
    orows = getindex!(rows_cache,cell_rows,cell)
    ocols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    rows = OIdsToIds(orows,row_terms)
    cols = OIdsToIds(ocols,col_terms)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end

@noinline function FESpaces._numeric_loop_vector!(vec,caches,cell_vals,cell_rows::OTable)
  add_cache,vals_cache,rows_cache = caches
  row_terms = cell_rows.terms
  add! = Arrays.AddEntriesMap(+)
  for cell in 1:length(cell_rows)
    orows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    rows = OIdsToIds(orows,row_terms)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end

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

  odof_to_dof = zeros(Int32,ndofs)
  for (inode,ipnode) in enumerate(node_to_pnode_linear)
    for icomp in b.dof_to_comp
      local_shift = (icomp-1)*nnodes
      odof_to_dof[local_shift+ipnode] = local_shift + inode
    end
  end

  return odof_to_dof
end
