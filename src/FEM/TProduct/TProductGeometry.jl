"""
    TProductModel{D,A,B} <: DiscreteModel{D,D} end

Tensor product discrete model, storing a vector of 1-D models `models_1d` of length D,
and the D-dimensional model `model` defined as their tensor product.
"""
struct TProductModel{D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}} <: DiscreteModel{D,D}
  model::A
  models_1d::B
end

Geometry.get_grid(model::TProductModel) = get_grid(model.model)
Geometry.get_grid_topology(model::TProductModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TProductModel) = get_face_labeling(model.model)

get_model(model::TProductModel) = model.model
get_1d_models(model::TProductModel) = model.models_1d

function _split_cartesian_descriptor(desc::CartesianDescriptor{D}) where D
  origin,sizes,partition,cmap,isperiodic = desc.origin,desc.sizes,desc.partition,desc.map,desc.isperiodic
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  descs = map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),isperiodic)
  return descs
end

function TProductModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = CartesianDiscreteModel.(descs_1d)
  TProductModel(model,models_1d)
end

function _d_to_lower_dim_entities(coords::AbstractArray{VectorValue{D,T},D}) where {D,T}
  entities = Vector{Array{VectorValue{D,T},D-1}}[]
  for d = 1:D
    range = axes(coords,d)
    bottom = selectdim(coords,d,first(range))
    top = selectdim(coords,d,last(range))
    push!(entities,[bottom,top])
  end
  return entities
end

_get_interior(entity::AbstractVector) = entity[2:end-1]
_get_interior(entity::AbstractMatrix) = entity[2:end-1,2:end-1]

function _tp_label_condition(intset,entity)
  interior = _get_interior(entity)
  for i in intset
    if i ∈ interior
      return false
    end
  end
  return true
end

"""
    get_1d_tags(model::TProductModel,tags) -> Vector{Vector{Int8}}

Fetches the tags of the tensor product 1D models corresponding to the tags
of the `D`-dimensional model `tags`. The length of the output is `D`
"""
function get_1d_tags(model::TProductModel{D},tags) where D
  isempty(tags) && return Vector{Vector{Int8}}(undef,D)

  nodes = get_node_coordinates(model)
  labeling = get_face_labeling(model)
  face_to_tag = get_face_tag_index(labeling,tags,0)

  d_to_entities = _d_to_lower_dim_entities(nodes)
  nodes_in_tag = nodes[findall(!iszero,face_to_tag)]

  msg = """
  The assigned boundary does not satisfy the tensor product condition:
  it should occupy the whole side of the domain, rather than a side's portion
  """

  map(1:D) do d
    tags = Int8[]
    entities = d_to_entities[d]
    for (tag1d,entity1d) in enumerate(entities)
      iset = intersect(nodes_in_tag,entity1d)
      if iset == vec(entity1d)
        push!(tags,tag1d)
      else
        @check _tp_label_condition(iset,entity1d) msg
      end
    end
    tags
  end
end

"""
    TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}

Tensor product triangulation, storing a tensor product model, a vector of 1-D
triangulations `trians_1d` of length D, and the D-dimensional triangulation `trian`
defined as their tensor product.
"""
struct TProductTriangulation{Dt,Dp,A<:TProductModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
end

function TProductTriangulation(trian::Triangulation,trians_1d::AbstractVector{<:Triangulation})
  model = get_background_model(trian)
  models_1d = map(get_background_model,trians_1d)
  tpmodel = TProductModel(model,models_1d)
  TProductTriangulation(tpmodel,trian,trians_1d)
end

Geometry.get_background_model(trian::TProductTriangulation) = trian.model
Geometry.get_grid(trian::TProductTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::TProductTriangulation{Dt},::Val{Dt}) where Dt = get_glue(trian.trian,Val{Dt}())

function Geometry.Triangulation(model::TProductModel;kwargs...)
  trian = Triangulation(model.model;kwargs...)
  trians_1d = map(Triangulation,model.models_1d)
  TProductTriangulation(model,trian,trians_1d)
end

for T in (:(AbstractVector{<:Integer}),:(AbstractVector{Bool}))
  @eval begin
    function Geometry.BoundaryTriangulation(
      model::TProductModel,
      face_to_bgface::$T,
      bgface_to_lcell::AbstractVector{<:Integer})

      BoundaryTriangulation(model.model,face_to_bgface,bgface_to_lcell)
    end
  end
end

function CellData.get_cell_points(trian::TProductTriangulation)
  point = get_cell_points(trian.trian)
  single_points = map(get_cell_points,trian.trians_1d)
  TProductCellPoint(point,single_points)
end

struct TProductMeasure{A,B} <: Measure
  measure::A
  measures_1d::B
end

function CellData.Measure(a::TProductTriangulation,args...;kwargs...)
  measure = Measure(a.trian,args...;kwargs...)
  measures_1d = map(Ω -> Measure(Ω,args...;kwargs...),a.trians_1d)
  TProductMeasure(measure,measures_1d)
end

function CellData.get_cell_points(a::TProductMeasure)
  point = get_cell_points(a.measure)
  single_points = map(get_cell_points,a.measures_1d)
  TProductCellPoint(point,single_points)
end

# default behavior

function CellData.integrate(f::CellField,b::TProductMeasure)
  integrate(f,b.measure)
end

# unfitted elements

function GridapEmbedded.cut(cutter::LevelSetCutter,background::TProductModel,geom)
  cut(cutter,background.model,geom)
end
