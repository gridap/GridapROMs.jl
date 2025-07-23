module ParamGeometry

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.Fields
using Gridap.Geometry
using Gridap.Helpers
using Gridap.ReferenceFEs

using GridapEmbedded
using GridapEmbedded.Interfaces
using GridapEmbedded.LevelSetCutters

using GridapROMs.Utils
using GridapROMs.ParamDataStructures

import FillArrays: Fill
import Gridap.Fields: AffineMap,ConstantMap

export ParamMappedGrid
export ParamMappedDiscreteModel
export ParamUnstructuredGrid

function _node_ids_to_coords!(node_ids_to_coords::GenericParamBlock,cell_node_ids,cell_to_coords)
  cache_node_ids = array_cache(cell_node_ids)
  cache_coords = array_cache(cell_to_coords)
  for k = 1:length(cell_node_ids)
    node_ids = getindex!(cache_node_ids,cell_node_ids,k)
    coords = getindex!(cache_coords,cell_to_coords,k)
    for j in param_eachindex(node_ids_to_coords)
      data = node_ids_to_coords.data[j]
      coord = coords[j]
      for (i,id) in enumerate(node_ids)
        data[node_ids[i]] = coord[i]
      end
    end
  end
end

struct ParamMappedGrid{Dc,Dp,T,M,L} <: Grid{Dc,Dp}
  grid::Grid{Dc,Dp}
  geo_map::T
  phys_map::M
  node_coords::L
end

function Geometry.MappedGrid(grid::Grid,phys_map::AbstractVector{<:ParamBlock})
  @assert length(phys_map) == num_cells(grid)
  plength = param_length(testitem(phys_map))

  function _compute_node_coordinates(grid,phys_map)
    cell_node_ids = get_cell_node_ids(grid)
    old_nodes = get_node_coordinates(grid)
    node_coordinates = Vector{eltype(old_nodes)}(undef,length(old_nodes))
    pnode_coordinates = parameterize(node_coordinates,plength)
    cell_to_coords = get_cell_coordinates(grid)
    cell_coords_map = lazy_map(evaluate,phys_map,cell_to_coords)
    _node_ids_to_coords!(pnode_coordinates,cell_node_ids,cell_coords_map)
    return pnode_coordinates
  end

  model_map = get_cell_map(grid)
  geo_map = lazy_map(∘,phys_map,model_map)
  node_coords = _compute_node_coordinates(grid,phys_map)
  ParamMappedGrid(grid,geo_map,phys_map,node_coords)
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

function Geometry.MappedDiscreteModel(model::DiscreteModel,phys_map::AbstractParamFunction)
  mapped_grid = MappedGrid(get_grid(model),phys_map)
  MappedDiscreteModel(model,mapped_grid)
end

Geometry.get_grid(model::ParamMappedDiscreteModel) = model.mapped_grid
Geometry.get_cell_map(model::ParamMappedDiscreteModel) = get_cell_map(model.mapped_grid)
Geometry.get_grid_topology(model::ParamMappedDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::ParamMappedDiscreteModel) = get_face_labeling(model.model)

function model_compatibility(a::DiscreteModel,b::DiscreteModel)
  return false
end

function model_compatibility(a::A,b::A) where {A<:DiscreteModel}
  return a === b
end

function model_compatibility(a::A,b::A) where {A<:Union{MappedDiscreteModel,ParamMappedDiscreteModel}}
  return a === b || a.model === b.model
end

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
  @check model_compatibility(smodel,tmodel)

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

function Fields.AffineField(gradients::ParamBlock,origins::ParamBlock)
  data = map(AffineField,get_param_data(gradients),get_param_data(origins))
  GenericParamBlock(data)
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  vi = return_value(k,gi,oi,x)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  gi = testitem(gradients)
  oi = testitem(origins)
  li = return_cache(k,gi,oi,x)
  vi = evaluate!(li,k,gi,oi,x)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x)

  @check param_length(gradients) == param_length(origins)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),x)
  end
  g
end

function Arrays.return_value(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  vi = return_value(k,gi,oi,xi)
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    g[i] = return_value(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(
  k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  gi = testitem(gradients)
  oi = testitem(origins)
  xi = testitem(x)
  li = return_cache(k,gi,oi,xi)
  vi = evaluate!(li,k,gi,oi,xi)
  l = Vector{typeof(li)}(undef,param_length(gradients))
  g = Vector{typeof(vi)}(undef,param_length(gradients))
  for i in param_eachindex(gradients)
    l[i] = return_cache(k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(
  cache,k::Broadcasting{<:AffineMap},gradients::ParamBlock,origins::ParamBlock,x::ParamBlock)

  @check param_length(gradients) == param_length(origins) == param_length(x)
  g,l = cache
  for i in param_eachindex(gradients)
    g.data[i] = evaluate!(l[i],k,param_getindex(gradients,i),param_getindex(origins,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  vi = return_value(k,ai,x)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  ai = testitem(a)
  li = return_cache(k,ai,x)
  vi = evaluate!(li,k,ai,x)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),x)
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),x)
  end
  g
end

function Arrays.return_value(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  vi = return_value(k,ai,xi)
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    g[i] = return_value(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  ai = testitem(a)
  xi = testitem(x)
  li = return_cache(k,ai,xi)
  vi = evaluate!(li,k,ai,xi)
  l = Vector{typeof(li)}(undef,param_length(a))
  g = Vector{typeof(vi)}(undef,param_length(a))
  for i in param_eachindex(a)
    l[i] = return_cache(k,param_getindex(a,i),param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,k::Broadcasting{<:ConstantMap},a::ParamBlock,x::ParamBlock)
  @check param_length(a) == param_length(x)
  g,l = cache
  for i in param_eachindex(a)
    g.data[i] = evaluate!(l[i],k,param_getindex(a,i),param_getindex(x,i))
  end
  g
end

function Arrays.return_value(f::Field,x::ParamBlock)
  xi = testitem(x)
  vi = return_value(f,xi)
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    g[i] = return_value(f,param_getindex(x,i))
  end
  GenericParamBlock(g)
end

function Arrays.return_cache(f::Field,x::ParamBlock)
  xi = testitem(x)
  li = return_cache(f,xi)
  vi = evaluate!(li,f,xi)
  l = Vector{typeof(li)}(undef,param_length(x))
  g = Vector{typeof(vi)}(undef,param_length(x))
  for i in param_eachindex(x)
    l[i] = return_cache(f,param_getindex(x,i))
  end
  GenericParamBlock(g),l
end

function Arrays.evaluate!(cache,f::Field,x::ParamBlock)
  g,l = cache
  for i in param_eachindex(x)
    g.data[i] = evaluate!(l[i],f,param_getindex(x,i))
  end
  g
end

for f in (:(Fields.GenericField),:(Fields.ZeroField),:(Fields.ConstantField))
  @eval begin
    $f(a::ParamBlock) = GenericParamBlock(map($f,get_param_data(a)))
  end
end

end
