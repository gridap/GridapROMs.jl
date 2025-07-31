abstract type GridMapStyle end
struct PhysicalMap <: GridMapStyle end
struct DisplacementMap <: GridMapStyle end

abstract type ParamGrid{Dc,Dp} <: Grid{Dc,Dp} end

struct ParamMappedGrid{Dc,Dp,A} <: ParamGrid{Dc,Dp}
  grid::Grid{Dc,Dp}
  node_coords::A
end

function _mapped_grid(style::GridMapStyle,grid::Grid,phys_map::AbstractVector)
  cell_node_ids = get_cell_node_ids(grid)
  old_nodes = get_node_coordinates(grid)
  node_coords = Vector{eltype(old_nodes)}(undef,length(old_nodes))
  _map_coords!(style,node_coords,old_nodes,cell_node_ids,phys_map)
  return VisMappedGrid(grid,node_coords)
end

function _mapped_grid(style::GridMapStyle,grid::Grid,phys_map::AbstractVector{<:GenericParamBlock})
  @assert length(phys_map) == num_cells(grid)
  plength = param_length(testitem(phys_map))

  cell_node_ids = get_cell_node_ids(grid)
  old_nodes = get_node_coordinates(grid)
  node_coords = Vector{eltype(old_nodes)}(undef,length(old_nodes))
  pnode_coords = parameterize(node_coords,plength)
  _map_coords!(style,pnode_coords,old_nodes,cell_node_ids,phys_map)

  return ParamMappedGrid(grid,pnode_coords)
end

function mapped_grid(style::GridMapStyle,grid::Grid,phys_map::AbstractVector)
  return _mapped_grid(style,grid,phys_map)
end

function mapped_grid(style::GridMapStyle,args...)
  @abstractmethod
end

function mapped_grid(grid::Grid,phys_map)
  mapped_grid(DisplacementMap(),grid,phys_map)
end

function mapped_grid(grid::Grid,phys_map::Function)
  mapped_grid(PhysicalMap(),grid,phys_map)
end

function mapped_grid(style::GridMapStyle,grid::Grid,phys_map::Function)
  cell_phys_map = Fill(GenericField(phys_map),num_cells(grid))
  cell_to_coords = get_cell_coordinates(grid)
  cell_coords_map = lazy_map(evaluate,cell_phys_map,cell_to_coords)
  mapped_grid(style,grid,cell_coords_map)
end

function mapped_grid(
  style::GridMapStyle,trian::BodyFittedTriangulation,phys_map::AbstractVector)
  model = get_background_model(trian)
  grid = mapped_grid(style,trian.grid,phys_map)
  BodyFittedTriangulation(model,grid,trian.tface_to_mface)
end

function mapped_grid(style::GridMapStyle,trian::Triangulation,φ::FEFunction)
  phys_map = φ(get_cell_points(trian))
  mapped_grid(style,trian,phys_map)
end

for T in (:FEFunction,:Function,:AbstractVector)
  @eval begin
    function mapped_grid(style::GridMapStyle,trian::Geometry.AppendedTriangulation,phys_map::$T)
      a = mapped_grid(style,trian.a,phys_map)
      b = mapped_grid(style,trian.b,phys_map)
      Geometry.AppendedTriangulation(a,b)
    end
  end
end

Geometry.get_node_coordinates(grid::ParamMappedGrid) = grid.node_coords
Geometry.get_cell_node_ids(grid::ParamMappedGrid) = get_cell_node_ids(grid.grid)
Geometry.get_reffes(grid::ParamMappedGrid) = get_reffes(grid.grid)
Geometry.get_cell_type(grid::ParamMappedGrid) = get_cell_type(grid.grid)

function Base.isapprox(t::T,s::T) where T<:ParamMappedGrid
  t.grid ≈ s.grid
end

function Base.isapprox(t::ParamMappedGrid,s::Grid)
  t.grid ≈ s
end

function Base.isapprox(t::Grid,s::ParamMappedGrid)
  t ≈ s.grid
end

"""
    struct ParamMappedDiscreteModel{Dc,Dp} <: DiscreteModel{Dc,Dp}
      model::DiscreteModel{Dc,Dp}
      mapped_grid::ParamMappedGrid{Dc,Dp}
    end

Represents a model with a `ParamMappedGrid` grid. See also `MappedDiscreteModel` in [`Gridap`](@ref).
"""
struct ParamMappedDiscreteModel{Dc,Dp} <: DiscreteModel{Dc,Dp}
  model::DiscreteModel{Dc,Dp}
  mapped_grid::ParamMappedGrid{Dc,Dp}
end

function Geometry.MappedDiscreteModel(model::DiscreteModel,mapped_grid::ParamMappedGrid)
  ParamMappedDiscreteModel(model,mapped_grid)
end

for T in (:AbstractParamFunction,:FEFunction)
  @eval begin
    function Geometry.MappedDiscreteModel(model::DiscreteModel,phys_map::$T)
      grid = mapped_grid(get_grid(model),phys_map)
      ParamMappedDiscreteModel(model,grid)
    end
  end
end

Geometry.get_grid(model::ParamMappedDiscreteModel) = model.mapped_grid
Geometry.get_cell_map(model::ParamMappedDiscreteModel) = get_cell_map(model.mapped_grid)
Geometry.get_grid_topology(model::ParamMappedDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::ParamMappedDiscreteModel) = get_face_labeling(model.model)

struct ParamUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: ParamGrid{Dc,Dp}
  node_coordinates::ParamBlock{Vector{Point{Dp,Tp}}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_types::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map

  function ParamUnstructuredGrid(
    node_coordinates::ParamBlock{Vector{Point{Dp,Tp}}},
    cell_node_ids::Table{Ti},
    reffes::Vector{<:LagrangianRefFE{Dc}},
    cell_types::Vector,
    orientation_style::OrientationStyle=NonOriented(),
    facet_normal=nothing;
    has_affine_map=nothing) where {Dc,Dp,Tp,Ti}

    if has_affine_map === nothing
      _has_affine_map = Geometry.get_has_affine_map(reffes)
    else
      _has_affine_map = has_affine_map
    end
    cell_map = Geometry._compute_cell_map(node_coordinates,cell_node_ids,reffes,cell_types,_has_affine_map)
    B = typeof(orientation_style)
    Tn = typeof(facet_normal)
    new{Dc,Dp,Tp,B,Tn}(
      node_coordinates,
      cell_node_ids,
      reffes,
      cell_types,
      orientation_style,
      facet_normal,
      cell_map)
  end
end

function Geometry.UnstructuredGrid(node_coordinates::ParamBlock{<:Vector{<:Point}},args...;kwargs...)
  ParamUnstructuredGrid(node_coordinates,args...;kwargs...)
end

function Geometry.UnstructuredGrid(grid::ParamUnstructuredGrid)
  grid
end

Geometry.OrientationStyle(
  ::Type{<:ParamUnstructuredGrid{Dc,Dp,Tp,B}}) where {Dc,Dp,Tp,B} = B()

Geometry.get_reffes(g::ParamUnstructuredGrid) = g.reffes
Geometry.get_cell_type(g::ParamUnstructuredGrid) = g.cell_types
Geometry.get_node_coordinates(g::ParamUnstructuredGrid) = g.node_coordinates
Geometry.get_cell_node_ids(g::ParamUnstructuredGrid) = g.cell_node_ids
Geometry.get_cell_map(g::ParamUnstructuredGrid) = g.cell_map

function Geometry.get_facet_normal(g::ParamUnstructuredGrid)
  @assert g.facet_normal != nothing "This Grid does not have information about normals."
  g.facet_normal
end

function Geometry.simplexify(grid::ParamUnstructuredGrid;kwargs...)
  reffes = get_reffes(grid)
  @notimplementedif length(reffes) != 1
  reffe = first(reffes)
  order = 1
  @notimplementedif get_order(reffe) != order
  p = get_polytope(reffe)
  ltcell_to_lpoints, simplex = simplexify(p;kwargs...)
  cell_to_points = get_cell_node_ids(grid)
  tcell_to_points = Geometry._refine_grid_connectivity(cell_to_points,ltcell_to_lpoints)
  ctype_to_reffe = [LagrangianRefFE(Float64,simplex,order),]
  tcell_to_ctype = fill(Int8(1),length(tcell_to_points))
  point_to_coords = get_node_coordinates(grid)
  ParamUnstructuredGrid(
    point_to_coords,
    tcell_to_points,
    ctype_to_reffe,
    tcell_to_ctype,
    Oriented())
end

function Base.isapprox(t::T,s::T) where T<:ParamUnstructuredGrid
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    length(testitem(t.node_coordinates)) == length(testitem(s.node_coordinates))
  )
end

function Base.isapprox(t::ParamUnstructuredGrid,s::UnstructuredGrid)
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    length(testitem(t.node_coordinates)) == length(s.node_coordinates)
  )
end

function Base.isapprox(t::UnstructuredGrid,s::ParamUnstructuredGrid)
  (
    t.cell_node_ids == s.cell_node_ids &&
    t.cell_types == s.cell_types &&
    length(t.node_coordinates) == length(testitem(s.node_coordinates))
  )
end

# for visualization, eventually should be moved to Gridap

struct VisMappedGrid{Dc,Dp,A} <: Grid{Dc,Dp}
  grid::Grid{Dc,Dp}
  node_coords::A
end

Geometry.get_node_coordinates(grid::VisMappedGrid) = grid.node_coords
Geometry.get_cell_node_ids(grid::VisMappedGrid) = get_cell_node_ids(grid.grid)
Geometry.get_reffes(grid::VisMappedGrid) = get_reffes(grid.grid)
Geometry.get_cell_type(grid::VisMappedGrid) = get_cell_type(grid.grid)

# utils

function _map_coords!(
  ::PhysicalMap,
  node_ids_to_coords,
  old_coords,
  cell_node_ids,
  cell_to_coords
  )

  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for (i,id) in enumerate(node_ids)
      node_ids_to_coords[id] = coords[i]
    end
  end
end

function _map_coords!(
  ::DisplacementMap,
  node_ids_to_coords,
  old_coords,
  cell_node_ids,
  cell_to_coords
  )

  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for (i,id) in enumerate(node_ids)
      node_ids_to_coords[id] = old_coords[id] + coords[i]
    end
  end
end

function _map_coords!(
  ::PhysicalMap,
  node_ids_to_coords::GenericParamBlock,
  old_coords,
  cell_node_ids,
  cell_to_coords::AbstractVector{<:GenericParamBlock}
  )

  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for j in param_eachindex(node_ids_to_coords)
      data = node_ids_to_coords.data[j]
      coord = coords.data[j]
      for (i,id) in enumerate(node_ids)
        data[id] = coord[i]
      end
    end
  end
end

function _map_coords!(
  ::DisplacementMap,
  node_ids_to_coords::GenericParamBlock,
  old_coords,
  cell_node_ids,
  cell_to_coords::AbstractVector{<:GenericParamBlock}
  )

  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for j in param_eachindex(node_ids_to_coords)
      data = node_ids_to_coords.data[j]
      coord = coords.data[j]
      for (i,id) in enumerate(node_ids)
        data[id] = old_coords[id] + coord[i]
      end
    end
  end
end

function _model_compatibility(a::DiscreteModel,b::DiscreteModel)
  return false
end

function _model_compatibility(a::A,b::A) where {A<:DiscreteModel}
  return a === b
end

function _model_compatibility(a::A,b::A) where {A<:Union{MappedDiscreteModel,ParamMappedDiscreteModel}}
  return a === b || a.model === b.model
end

function _prepare_node_to_coords(cell_to_points::AbstractVector{<:ParamBlock})
  _prepare_node_to_coords(lazy_testitem(cell_to_points))
end
