for T in (:GenericParamBlock,:Field)
  @eval begin
    function mapped_grid(
      style::GridMapStyle,trian::Interfaces.ParamSubFacetTriangulation,phys_map::AbstractVector{<:$T})
      model = get_background_model(trian)
      subgrid = mapped_grid(trian.subgrid,phys_map)
      subfacets = _change_coords(trian.subfacets,subgrid)
      ParamSubFacetTriangulation(subfacets,model)
    end
  end
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
  reffe = LagrangianRefFE(Float64,Simplex(Val{Dc}()),1)
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

A triangulation for subfacets.
"""
struct ParamSubFacetTriangulation{Dc,Dp,T,A} <: Triangulation{Dc,Dp}
  subfacets::ParamSubFacetData{Dp,T}
  bgmodel::A
  subgrid::UnstructuredGrid{Dc,Dp,T,NonOriented,Nothing}
  function ParamSubFacetTriangulation(
    subfacets::ParamSubFacetData{Dp,T},bgmodel::DiscreteModel) where {Dp,T}
    Dc = Dp-1
    subgrid = UnstructuredGrid(subfacets)
    A = typeof(bgmodel)
    new{Dc,Dp,T,A}(subfacets,bgmodel,subgrid)
  end
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

function Geometry.compute_active_model(trian::ParamSubFacetTriangulation)
  subgrid = trian.subgrid
  subfacets = trian.subfacets
  facet_to_uids,uid_to_point = consistent_facet_to_points(
    subfacets.facet_to_points,subfacets.point_to_coords
  )
  error("Must implement an unstructured grid topology for param geometries!")
  topo = UnstructuredGridTopology(
    subgrid,facet_to_uids,uid_to_point
  )
  return UnstructuredDiscreteModel(subgrid,topo,FaceLabeling(topo))
end

function Interfaces.consistent_facet_to_points(
  facet_to_points::Table,
  point_to_coords::ParamBlock
  )

  f(pt::VectorValue) = VectorValue(round.(pt.data;sigdigits=12))
  f(pt::ParamBlock) = ParamBlock(map(f,pt.data))
  f(id::Integer) = f(point_to_coords[id])

  # Create a list of the unique points composing the facets
  npts = length(point_to_coords)
  nfaces = length(facet_to_points)
  touched = zeros(Bool,npts)
  for face in 1:nfaces
    pts = view(facet_to_points,face)
    touched[pts] .= true
  end
  touched_ids = findall(touched)
  unique_ids = unique(f,touched_ids)

  # Create a mapping from the old point ids to the new ones
  touched_to_uid = collect(Int32,indexin(f.(touched_ids),f.(unique_ids)))
  point_to_uid = extend(touched_to_uid,PosNegPartition(touched_ids,npts))

  facet_to_uids = Table(
    collect(Int32,lazy_map(Reindex(point_to_uid),facet_to_points.data)),
    facet_to_points.ptrs
  )
  return facet_to_uids,unique_ids
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
