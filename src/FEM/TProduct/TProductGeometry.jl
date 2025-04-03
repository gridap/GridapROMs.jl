"""
    TProductDiscreteModel{D,A,B} <: DiscreteModel{D,D} end

Tensor product discrete model, storing a vector of 1-D models `models_1d` of length D,
and the D-dimensional model `model` defined as their tensor product.
"""
struct TProductDiscreteModel{D,A<:CartesianDiscreteModel{D},B<:AbstractVector{<:CartesianDiscreteModel}} <: DiscreteModel{D,D}
  model::A
  models_1d::B
end

Geometry.get_grid(model::TProductDiscreteModel) = get_grid(model.model)
Geometry.get_grid_topology(model::TProductDiscreteModel) = get_grid_topology(model.model)
Geometry.get_face_labeling(model::TProductDiscreteModel) = get_face_labeling(model.model)

get_model(model::TProductDiscreteModel) = model.model
get_1d_models(model::TProductDiscreteModel) = model.models_1d

function _split_cartesian_descriptor(desc::CartesianDescriptor{D}) where D
  origin,sizes,partition,cmap,isperiodic = desc.origin,desc.sizes,desc.partition,desc.map,desc.isperiodic
  function _compute_1d_desc(
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic))
    CartesianDescriptor(Point(o),(s,),(p,);map=m,isperiodic=(i,))
  end
  descs = map(_compute_1d_desc,origin.data,sizes,partition,Fill(cmap,D),isperiodic)
  return descs
end

function TProductDiscreteModel(args...;kwargs...)
  desc = CartesianDescriptor(args...;kwargs...)
  descs_1d = _split_cartesian_descriptor(desc)
  model = CartesianDiscreteModel(desc)
  models_1d = CartesianDiscreteModel.(descs_1d)
  TProductDiscreteModel(model,models_1d)
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

function _throw_tp_error()
  msg = """
  The assigned boundary does not satisfy the tensor product condition:
  it should occupy the whole side of the domain, rather than a side's portion. Try
  imposing the Dirichlet condition weakly, e.g. with Nitsche's penalty method
  """

  @assert false msg
end

function _check_tp_label_condition(intset,entity)
  interior = _get_interior(entity)
  for i in intset
    if i ∈ interior
      return _throw_tp_error()
    end
  end
  return true
end

"""
    get_1d_tags(model::TProductDiscreteModel,tags) -> Vector{Vector{Int8}}

Fetches the tags of the tensor product 1D models corresponding to the tags
of the `D`-dimensional model `tags`. The length of the output is `D`
"""
function get_1d_tags(model::TProductDiscreteModel{D},tags) where D
  nodes = get_node_coordinates(model)
  labeling = get_face_labeling(model)
  face_to_tag = get_face_tag_index(labeling,tags,0)

  d_to_entities = _d_to_lower_dim_entities(nodes)
  nodes_in_tag = nodes[findall(!iszero,face_to_tag)]

  map(1:D) do d
    d_tags = Int8[]
    if !isempty(tags)
      entities = d_to_entities[d]
      for (tag1d,entity1d) in enumerate(entities)
        iset = intersect(nodes_in_tag,entity1d)
        if iset == vec(entity1d)
          push!(d_tags,tag1d)
        else
          _check_tp_label_condition(iset,entity1d)
        end
      end
    end
    d_tags
  end
end

"""
    TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}

Tensor product triangulation, storing a tensor product model, a vector of 1-D
triangulations `trians_1d` of length D, and the D-dimensional triangulation `trian`
defined as their tensor product.
"""
struct TProductTriangulation{Dt,Dp,A<:TProductDiscreteModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
end

function TProductTriangulation(trian::Triangulation,trians_1d::AbstractVector{<:Triangulation})
  model = get_background_model(trian)
  models_1d = map(get_background_model,trians_1d)
  tpmodel = TProductDiscreteModel(model,models_1d)
  TProductTriangulation(tpmodel,trian,trians_1d)
end

Base.:(==)(a::TProductTriangulation,b::TProductTriangulation) = a.trian == b.trian
Geometry.get_background_model(trian::TProductTriangulation) = trian.model
Geometry.get_grid(trian::TProductTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::TProductTriangulation{Dt},::Val{Dt}) where Dt = get_glue(trian.trian,Val{Dt}())

function Geometry.Triangulation(model::TProductDiscreteModel;kwargs...)
  trian = Triangulation(model.model;kwargs...)
  trians_1d = map(Triangulation,model.models_1d)
  TProductTriangulation(model,trian,trians_1d)
end

for T in (:(AbstractVector{<:Integer}),:(AbstractVector{Bool}))
  @eval begin
    function Geometry.BoundaryTriangulation(
      model::TProductDiscreteModel,
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

"""
    struct TProductMeasure{A,B} <: Measure
      measure::A
      measures_1d::B
    end

Tensor product measure, storing a vector of 1-D measures `measures_1d` of length D,
and the D-dimensional measure `measure` defined as their tensor product.
"""
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

function GridapEmbedded.cut(cutter::LevelSetCutter,background::TProductDiscreteModel,geom)
  cut(cutter,background.model,geom)
end
