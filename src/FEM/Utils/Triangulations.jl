"""
    is_parent(parent::Triangulation,child::Triangulation) -> Bool

Returns true if `child` is a triangulation view of `parent`, false otherwise
"""
function is_parent(parent::Triangulation,child::Triangulation)
  false
end

function is_parent(
  parent::BodyFittedTriangulation,
  child::BodyFittedTriangulation{Dt,Dp,A,<:Geometry.GridView}) where {Dt,Dp,A}
  parent.grid === child.grid.parent
end

function is_parent(parent::Triangulation,child::Geometry.TriangulationView)
  parent === child.parent
end

for T in (:Triangulation,:(BodyFittedTriangulation{Dt,Dp,A,<:Geometry.GridView} where {Dt,Dp,A}),:(Geometry.TriangulationView))
  @eval begin
    function is_parent(parent::Geometry.AppendedTriangulation,child::$T)
      is_parent(parent.a,child) || is_parent(parent.b,child)
    end
  end
end

function is_parent(parent::Geometry.AppendedTriangulation,child::Geometry.AppendedTriangulation)
  is_parent(parent.a,child.a) && is_parent(parent.b,child.b)
end

function is_parent(parent::SkeletonTriangulation,child::SkeletonPair)
  is_parent(parent.plus,child.plus) && is_parent(parent.minus,child.minus)
end

function get_parent(i::AbstractVector)
  i
end

function get_parent(i::LazyArray{<:Fill{<:Reindex}})
  i.maps.value.values
end

function get_parent(t::Geometry.Grid)
  t
end

function get_parent(gv::Geometry.GridView)
  gv.parent
end

function get_parent(t::Geometry.TriangulationView)
  t.parent
end

function get_parent(t::BodyFittedTriangulation)
  grid = get_parent(get_grid(t))
  model = get_background_model(t)
  tface_to_mface = get_parent(t.tface_to_mface)
  BodyFittedTriangulation(model,grid,tface_to_mface)
end

function get_parent(t::Geometry.AppendedTriangulation)
  a = get_parent(t.a)
  b = get_parent(t.b)
  lazy_append(a,b)
end

"""
    get_parent(t::Triangulation) -> Triangulation

When `t` is a triangulation view, returns its parent; throws an error when `t`
is not a triangulation view
"""
function get_parent(t::Triangulation)
  t
end

_get_bg_cells(t::Triangulation) = @abstractmethod
_get_bg_cells(t::Geometry.BodyFittedTriangulation) = t.tface_to_mface
_get_bg_cells(t::Geometry.BoundaryTriangulation) = t.glue.face_to_cell
_get_bg_cells(t::Geometry.TriangulationView) = t.cell_to_parent_cell
_get_bg_cells(t::Interfaces.SubFacetTriangulation) = t.subfacets.facet_to_bgcell
_get_bg_cells(t::Interfaces.SubCellTriangulation) = unique(t.subcells.cell_to_bgcell)
function _get_bg_cells(t::Geometry.AppendedTriangulation)
  lazy_append(_get_bg_cells(t.a),_get_bg_cells(t.b))
end

function strian_to_ttrian_cells(strian::Triangulation,ttrian::Triangulation)
  scells = _get_bg_cells(strian)
  tcells = _get_bg_cells(ttrian)
  isentry = zeros(Bool,length(scells))
  for i in eachindex(scells)
    isentry[i] = scells[i] ≤ length(tcells)
  end
  entries = cumsum(isentry)
  s2t = Vector{eltype(tcells)}(undef,entries[end])
  for i in eachindex(scells)
    if isentry[i]
      s2t[entries[i]] = tcells[scells[i]]
    end
  end
  return s2t
end

# We use the symbol ≈ can between two grids `t` and `s` in the following
# sense: if `t` and `s` store the same information but have a different UID, then
# this function returns true; otherwise, it returns false

function Base.:(==)(a::Geometry.CartesianCoordinates,b::Geometry.CartesianCoordinates)
  false
end

function Base.:(==)(a::T,b::T) where T<:Geometry.CartesianCoordinates
  (
    a.data.origin == b.data.origin &&
    a.data.sizes == b.data.sizes &&
    a.data.partition == b.data.partition
  )
end

function Base.:(==)(a::Geometry.CartesianCellNodes,b::Geometry.CartesianCellNodes)
  a.partition == b.partition
end

function Base.:(==)(a::Table,b::Table)
  a.data == b.data && a.ptrs == b.ptrs
end

function Base.isapprox(t::Grid,s::Grid)
  false
end

function Base.isapprox(t::T,s::T) where T<:Grid
  @notimplemented "Implementation needed"
end

function Base.isapprox(t::T,s::T) where T<:Geometry.CartesianGrid
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_type == s.cell_type &&
    t.node_coords == s.node_coords
  )
end

function Base.isapprox(t::T,s::T) where T<:Geometry.GridPortion
  (
    t.parent ≈ s.parent &&
    t.cell_to_parent_cell == s.cell_to_parent_cell &&
    t.node_to_parent_node == s.node_to_parent_node &&
    t.cell_to_nodes == s.cell_to_nodes
  )
end

function Base.isapprox(t::T,s::T) where T<:Geometry.GridView
  t.parent ≈ s.parent && t.cell_to_parent_cell == s.cell_to_parent_cell
end

function Base.isapprox(t::T,s::T) where T<:UnstructuredGrid
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    t.node_coordinates == s.node_coordinates
  )
end

function Base.isapprox(t::T,s::T) where T<:Geometry.AppendedGrid
  t.a ≈ s.a && t.b ≈ s.b
end

function Base.isapprox(t::T,s::T) where T<:Triangulation
  get_grid(t) ≈ get_grid(s)
end

function find_trian_permutation(a,b,cmp::Function)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

function find_trian_permutation(a,b)
  cmp = (a,b) -> a == b || is_parent(a,b)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

"""
    order_domains(
      parents::Tuple{Vararg{Triangulation}},
      children::Tuple{Vararg{Triangulation}}
      ) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents
"""
function order_domains(parents,children)
  @check length(parents) == length(children)
  iperm = find_trian_permutation(parents,children)
  map(iperm->children[iperm],iperm)
end

# triangulation views

function FESpaces.get_cell_fe_data(fun,f,ttrian::Geometry.TriangulationView)
  parent_vals = FESpaces.get_cell_fe_data(fun,f,ttrian.parent)
  return lazy_map(Reindex(parent_vals),ttrian.cell_to_parent_cell)
end

@inline function Geometry.is_change_possible(strian::Geometry.TriangulationView,ttrian::Triangulation)
  return false
end

@inline function Geometry.is_change_possible(strian::Triangulation,ttrian::Geometry.TriangulationView)
  return Geometry.is_change_possible(strian,ttrian.parent)
end

function CellData.change_domain(a::CellField,strian::Triangulation,::ReferenceDomain,ttrian::Geometry.TriangulationView,::ReferenceDomain)
  if strian === ttrian
    return a
  end
  parent = change_domain(a,strian,ReferenceDomain(),ttrian.parent,ReferenceDomain())
  cell_data = lazy_map(Reindex(get_data(parent)),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,ReferenceDomain())
end

function CellData.change_domain(a::CellField,strian::Triangulation,::PhysicalDomain,ttrian::Geometry.TriangulationView,::PhysicalDomain)
  if strian === ttrian
    return a
  end
  parent = change_domain(a,strian,PhysicalDomain(),ttrian.parent,PhysicalDomain())
  cell_data = lazy_map(Reindex(get_data(parent)),ttrian.cell_to_parent_cell)
  return CellData.similar_cell_field(a,cell_data,ttrian,PhysicalDomain())
end

# correct bug

function Base.view(trian::Geometry.AppendedTriangulation,ids::AbstractArray)
  Ti = eltype(ids)
  ids1 = Ti[]
  ids2 = Ti[]
  n1 = num_cells(trian.a)
  for i in ids
    if i <= n1
      push!(ids1,i)
    else
      push!(ids2,i-n1)
    end
  end
  trian1 = view(trian.a,ids1)
  trian2 = view(trian.b,ids2)
  isempty(ids1) ? trian2 : (isempty(ids2) ? trian1 : lazy_append(trian1,trian2))
end
