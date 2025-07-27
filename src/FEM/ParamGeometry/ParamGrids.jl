
abstract type GridMapStyle end
struct PhysicalMap <: GridMapStyle end
struct DisplacementMap <: GridMapStyle end

struct ParamMappedGrid{Dc,Dp,A} <: Grid{Dc,Dp}
  grid::Grid{Dc,Dp}
  node_coords::A
end

function mapped_grid(style::GridMapStyle,grid::Grid,phys_map::AbstractVector{<:GenericParamBlock})
  @assert length(phys_map) == num_cells(grid)
  plength = param_length(testitem(phys_map))

  cell_node_ids = get_cell_node_ids(grid)
  old_nodes = get_node_coordinates(grid)
  node_coords = Vector{eltype(old_nodes)}(undef,length(old_nodes))
  pnode_coords = parameterize(node_coords,plength)
  _map_coords!(style,pnode_coords,old_nodes,cell_node_ids,phys_map)

  return ParamMappedGrid(grid,pnode_coords)
end

function mapped_grid(style::GridMapStyle,args...)
  @abstractmethod
end

function mapped_grid(grid::Grid,phys_map)
  mapped_grid(DisplacementMap(),grid,phys_map)
end

function mapped_grid(grid::Grid,phys_map::AbstractParamFunction)
  mapped_grid(PhysicalMap(),grid,phys_map)
end

function mapped_grid(style::GridMapStyle,grid::Grid,phys_map::AbstractVector{<:Field})
  cell_to_coords = get_cell_coordinates(grid)
  cell_coords_map = lazy_map(evaluate,phys_map,cell_to_coords)
  mapped_grid(style,grid,phys_map)
end

function mapped_grid(style::GridMapStyle,grid::Grid,phys_map::Function)
  cell_phys_map = Fill(GenericField(phys_map),num_cells(grid))
  mapped_grid(style,grid,cell_phys_map)
end

for T in (:GenericParamBlock,:Field)
  @eval begin
    function mapped_grid(
      style::GridMapStyle,trian::BodyFittedTriangulation,phys_map::AbstractVector{<:$T})
      model = get_background_model(trian)
      grid = mapped_grid(style,trian.grid,phys_map)
      BodyFittedTriangulation(model,grid,trian.tface_to_mface)
    end
  end
end

for T in (:FEFunction,:Function,:(AbstractVector{<:GenericParamBlock}),:(AbstractVector{<:Field}))
  @eval begin
    function mapped_grid(style::GridMapStyle,trian::Geometry.AppendedTriangulation,phys_map::$T)
      a = mapped_grid(style,trian.a,phys_map)
      b = mapped_grid(style,trian.b,phys_map)
      Geometry.AppendedTriangulation(a,b)
    end
  end
end

function mapped_grid(style::GridMapStyle,trian::Triangulation,φ::FEFunction)
  phys_map = φ(get_cell_points(trian))
  mapped_grid(style,trian,phys_map)
end

Geometry.get_node_coordinates(grid::ParamMappedGrid) = grid.node_coords
Geometry.get_cell_node_ids(grid::ParamMappedGrid) = get_cell_node_ids(grid.grid)
Geometry.get_reffes(grid::ParamMappedGrid) = get_reffes(grid.grid)
Geometry.get_cell_type(grid::ParamMappedGrid) = get_cell_type(grid.grid)

"""
MappedDiscreteModel

Represent a model with a `MappedGrid` grid.
See also [`MappedGrid`](@ref).
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
      mapped_grid = mapped_grid(get_grid(model),phys_map)
      ParamMappedDiscreteModel(model,mapped_grid)
    end
  end
end

Geometry.get_grid(model::ParamMappedDiscreteModel) = model.mapped_grid
Geometry.get_cell_map(model::ParamMappedDiscreteModel) = get_cell_map(model.mapped_grid)
Geometry.get_grid_topology(model::ParamMappedDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::ParamMappedDiscreteModel) = get_face_labeling(model.model)

function Geometry.is_change_possible(
  strian::Triangulation{Dc,Dp},
  ttrian::Triangulation{Dc,Dp}
  ) where {Dc,Dp}

  if strian === ttrian
    return true
  end

  msg = "Triangulations do not point to the same background discrete model!"
  smodel = get_background_model(strian)
  tmodel = get_background_model(strian)
  @check _model_compatibility(smodel,tmodel)

  sglue = get_glue(strian,Val(Dc))
  tglue = get_glue(ttrian,Val(Dc))
  is_change_possible(sglue,tglue)
end

function Geometry.Grid(::Type{ReferenceFE{d}},model::ParamMappedDiscreteModel) where d
  node_coordinates = get_node_coordinates(model)
  cell_to_nodes = Table(get_face_nodes(model,d))
  cell_to_type = collect1d(get_face_type(model,d))
  reffes = get_reffaces(ReferenceFE{d},model)
  UnstructuredGrid(node_coordinates,cell_to_nodes,reffes,cell_to_type)
end

struct ParamUnstructuredGrid{Dc,Dp,Tp,O,Tn} <: Grid{Dc,Dp}
  node_coordinates::ParamBlock{Vector{Point{Dp,Tp}}}
  cell_node_ids::Table{Int32,Vector{Int32},Vector{Int32}}
  reffes::Vector{LagrangianRefFE{Dc}}
  cell_type::Vector{Int8}
  orientation_style::O
  facet_normal::Tn
  cell_map

  function ParamUnstructuredGrid(
    node_coordinates::ParamBlock{Vector{Point{Dp,Tp}}},
    cell_node_ids::Table{Ti},
    reffes::Vector{<:LagrangianRefFE{Dc}},
    cell_type::Vector,
    orientation_style::OrientationStyle=NonOriented(),
    facet_normal=nothing;
    has_affine_map=nothing) where {Dc,Dp,Tp,Ti}

    if has_affine_map === nothing
      _has_affine_map = Geometry.get_has_affine_map(reffes)
    else
      _has_affine_map = has_affine_map
    end
    cell_map = Geometry._compute_cell_map(node_coordinates,cell_node_ids,reffes,cell_type,_has_affine_map)
    B = typeof(orientation_style)
    Tn = typeof(facet_normal)
    new{Dc,Dp,Tp,B,Tn}(
      node_coordinates,
      cell_node_ids,
      reffes,
      cell_type,
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
Geometry.get_cell_type(g::ParamUnstructuredGrid) = g.cell_type
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
    t.cell_type == s.cell_type &&
    length(testitem(t.node_coordinates)) == length(testitem(s.node_coordinates))
  )
end

# utils

function _map_coords!(
  ::PhysicalMap,
  old_coords,
  node_ids_to_coords::GenericParamBlock,
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
