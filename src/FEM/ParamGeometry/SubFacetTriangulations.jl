function mapped_grid(
  style::GridMapStyle,
  trian::Interfaces.SubFacetTriangulation,
  phys_map::AbstractVector
  )

  model = get_background_model(trian)
  subgrid = mapped_grid(trian.subgrid,phys_map)
  subfacets = _change_coords(trian.subfacets,subgrid)
  Interfaces.SubFacetTriangulation(subfacets,model)
end

struct ParamSubFacetData{Dp,T,Tn} <: GridapType
  facet_to_points::Table{Int32,Vector{Int32},Vector{Int32}}
  facet_to_normal::Vector{Point{Dp,Tn}}
  facet_to_bgcell::Vector{Int32}
  point_to_coords::ParamBlock{Vector{Point{Dp,T}}}
  point_to_rcoords::Vector{Point{Dp,T}}
end

function Interfaces.SubFacetData(
  facet_to_points::Table,
  facet_to_normal::AbstractVector,
  facet_to_bgcell::AbstractVector,
  point_to_coords::ParamBlock,
  point_to_rcoords::AbstractVector
  )

  ParamSubFacetData(
    facet_to_points,
    facet_to_normal,
    facet_to_bgcell,
    point_to_coords,
    point_to_rcoords)
end

function Geometry.UnstructuredGrid(st::ParamSubFacetData{Dp}) where Dp
  Dc = Dp - 1
  reffe = LagrangianRefFE(Float64,Interfaces.Simplex(Val{Dc}()),1)
  cell_types = fill(Int8(1),length(st.facet_to_points))
  UnstructuredGrid(
    st.point_to_coords,
    st.facet_to_points,
    [reffe,],
    cell_types)
end

function Base.empty(st::ParamSubFacetData{Dp,T,Tn}) where {Dp,T,Tn}
  plength = param_length(st.point_to_coords)
  facet_to_points = Table(Int32[],Int32[1,])
  facet_to_normal = Point{Dp,Tn}[]
  facet_to_bgcell = Int32[]
  point_to_coords = parameterize(Point{Dp,T}[],plength)
  point_to_rcoords = Point{Dp,T}[]

  ParamSubFacetData(
    facet_to_points,
    facet_to_normal,
    facet_to_bgcell,
    point_to_coords,
    point_to_rcoords)
end

function Base.append!(a::ParamSubFacetData{D},b::ParamSubFacetData{D}) where D
  @check param_length(a.point_to_coords)==param_length(b.point_to_coords)
  o = length(a.point_to_coords)

  append!(a.facet_to_normal,b.facet_to_normal)
  append!(a.facet_to_bgcell,b.facet_to_bgcell)
  for i in param_eachindex(a.point_to_coords)
    append!(a.point_to_coords.data[i],b.point_to_coords.data[i])
  end
  append!(a.point_to_rcoords,b.point_to_rcoords)

  nini = length(a.facet_to_points.data)+1
  append!(a.facet_to_points.data,b.facet_to_points.data)
  nend = length(a.facet_to_points.data)
  append_ptrs!(a.facet_to_points.ptrs,b.facet_to_points.ptrs)
  for i in nini:nend
    a.facet_to_points.data[i] += o
  end

  a
end

# Implementation of the Gridap.Triangulation interface

"""
    struct ParamSubFacetTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
      subfacets::ParamSubFacetData{Dp,T}
      bgmodel::A
      subgrid::ParamUnstructuredGrid{Dc,Dp,T,NonOriented,Nothing}
    end

Parameterized version of a [`SubFacetTriangulation`](@ref) in [`GridapEmbedded`](@ref)
"""
struct ParamSubFacetTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
  subfacets::ParamSubFacetData{Dp,T}
  bgmodel::A
  subgrid::ParamUnstructuredGrid{Dc,Dp,T,NonOriented,Nothing}
  function ParamSubFacetTriangulation(
    subfacets::ParamSubFacetData{Dp,T},bgmodel::DiscreteModel) where {Dp,T}
    Dc = Dp-1
    subgrid = UnstructuredGrid(subfacets)
    A = typeof(bgmodel)
    new{Dc,Dp,T,A}(subfacets,bgmodel,subgrid)
  end
end

function Interfaces.SubFacetTriangulation(subfacets::ParamSubFacetData,model::DiscreteModel)
  ParamSubFacetTriangulation(subfacets,model)
end

function Geometry.get_background_model(a::ParamSubFacetTriangulation)
  a.bgmodel
end

function Geometry.get_grid(a::ParamSubFacetTriangulation)
  a.subgrid
end

function Geometry.get_glue(a::ParamSubFacetTriangulation{Dc},::Val{D}) where {Dc,D}
  if (D-1) != Dc
    return nothing
  end
  tface_to_mface = a.subfacets.facet_to_bgcell
  tface_to_mface_map = Interfaces._setup_facet_ref_map(a.subfacets,a.subgrid)
  FaceToFaceGlue(tface_to_mface,tface_to_mface_map,nothing)
end

function Geometry.get_facet_normal(a::ParamSubFacetTriangulation)
  lazy_map(constant_field,a.subfacets.facet_to_normal)
end

function Geometry.move_contributions(scell_to_val::AbstractArray,strian::ParamSubFacetTriangulation)
  model = get_background_model(strian)
  ncells = num_cells(model)
  cell_to_touched = fill(false,ncells)
  scell_to_cell = strian.subfacets.facet_to_bgcell
  cell_to_touched[scell_to_cell] .= true
  Ωa = Triangulation(model,cell_to_touched)
  acell_to_val = move_contributions(scell_to_val,strian,Ωa)
  acell_to_val, Ωa
end

function Geometry.get_active_model(trian::ParamSubFacetTriangulation)
  msg = """
  This is not implemented, but also not needed in practice.
  Embedded Grids implemented for integration, not interpolation.
  """
  @notimplemented  msg
end

# utils

function _change_coords(sf::Interfaces.SubFacetData,grid::Grid)
  Interfaces.SubFacetData(
    sf.facet_to_points,
    sf.facet_to_normal,
    sf.facet_to_bgcell,
    get_node_coordinates(grid),
    sf.point_to_rcoords
    )
end

function Base.isapprox(t::T,s::T) where T<:ParamSubFacetData
  (
    t.facet_to_points == s.facet_to_points &&
    t.facet_to_normal == s.facet_to_normal &&
    t.facet_to_bgcell == s.facet_to_bgcell &&
    length(testitem(t.point_to_coords)) == length(testitem(s.point_to_coords)) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Base.isapprox(t::ParamSubFacetData,s::Interfaces.SubFacetData)
  (
    t.facet_to_points == s.facet_to_points &&
    t.facet_to_normal == s.facet_to_normal &&
    t.facet_to_bgcell == s.facet_to_bgcell &&
    length(testitem(t.point_to_coords)) == length(s.point_to_coords) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Base.isapprox(t::Interfaces.SubFacetData,s::ParamSubFacetData)
  (
    t.facet_to_points == s.facet_to_points &&
    t.facet_to_normal == s.facet_to_normal &&
    t.facet_to_bgcell == s.facet_to_bgcell &&
    length(t.point_to_coords) == length(testitem(s.point_to_coords)) &&
    t.point_to_rcoords == s.point_to_rcoords
  )
end

function Utils.to_child(parent::ParamSubFacetTriangulation,child::Geometry.TriangulationView)
  @check isa(child.parent,ParamSubFacetTriangulation)
  Geometry.TriangulationView(parent,child.cell_to_parent_cell)
end
