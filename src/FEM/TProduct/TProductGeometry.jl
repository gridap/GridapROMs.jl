"""
    struct TProductDiscreteModel{D,A,B} <: DiscreteModel{D,D}
      model::A
      models_1d::B
    end

A `D`-dimensional `CartesianDiscreteModel` together with `D` 1D
`CartesianDiscreteModel`s whose Cartesian product reproduces it.

Use [`TProductTriangulation`](@ref) and [`TProductMeasure`](@ref) to build
the corresponding integration objects, and [`TProductFESpace`](@ref) (or the
[`TensorProductReferenceFE`](@ref) interface) to build the FE space.

# Construction

    TProductDiscreteModel(args...; kwargs...)

Accepts the same arguments as `CartesianDiscreteModel`: a domain tuple and a
partition tuple. The 1D components are split automatically from the
D-dimensional `CartesianDescriptor`.

# Example

```julia
model = TProductDiscreteModel((0,1,0,1),(10,10))  # 10×10 Cartesian mesh on [0,1]²
```
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
    o=first(origin.data),s=first(sizes),p=first(partition),m=cmap,i=first(isperiodic)
    )
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
    struct TProductTriangulation{Dt,Dp,A,B,C} <: Triangulation{Dt,Dp}
      model::A
      trian::B
      trians_1d::C
    end

A `Triangulation` whose cells are the Cartesian product of `D` 1D
triangulations stored in `trians_1d`. The full D-dimensional triangulation
`trian` and the background [`TProductDiscreteModel`](@ref) `model` are also
stored for standard Gridap compatibility.

Construct via `Triangulation(model::TProductDiscreteModel)` or by wrapping an
existing `Triangulation` with a vector of 1D triangulations.
"""
struct TProductTriangulation{Dt,Dp,A<:TProductDiscreteModel,B<:BodyFittedTriangulation{Dt,Dp},C<:AbstractVector{<:Triangulation}} <: Triangulation{Dt,Dp}
  model::A
  trian::B
  trians_1d::C
end

function TProductTriangulation(
  trian::Triangulation,
  trians_1d::AbstractVector{<:Triangulation}
  )
  model = get_background_model(trian)
  models_1d = map(get_background_model,trians_1d)
  tpmodel = TProductDiscreteModel(model,models_1d)
  TProductTriangulation(tpmodel,trian,trians_1d)
end

Base.:(==)(a::TProductTriangulation,b::TProductTriangulation) = a.trian == b.trian
Geometry.get_background_model(trian::TProductTriangulation) = trian.model
Geometry.get_grid(trian::TProductTriangulation) = get_grid(trian.trian)
Geometry.get_glue(trian::TProductTriangulation{Dt},::Val{Dt}) where Dt = get_glue(trian.trian,Val{Dt}())
Utils.filter_domains(trian::TProductTriangulation) = trian.trian

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
      bgface_to_lcell::AbstractVector{<:Integer}
      )

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

A `Measure` whose quadrature is the Cartesian product of `D` 1D quadratures
stored in `measures_1d`. The full D-dimensional measure `measure` is also kept
for standard Gridap integration.

Integrating a [`TProductCellField`](@ref) against a `TProductMeasure` returns
a vector of `D` `DomainContribution`s, one per spatial direction, which are
later assembled into a [`AbstractRankTensor`](@ref) by
[`TProductSparseMatrixAssembler`](@ref).

Construct via `Measure(trian::TProductTriangulation,degree;kwargs...)`.
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
