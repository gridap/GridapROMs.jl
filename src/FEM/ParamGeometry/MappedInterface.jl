struct MappedMeasure <: Measure
  quad::CellQuadrature
  cell_map
end

function MappedMeasure(ref_quad::CellQuadrature,t::Triangulation)
  ref_trian = ref_quad.trian
  cell_map_on_ref_trian = _get_cell_map_on_trian(t,ref_quad.trian)
  MappedMeasure(ref_quad,cell_map_on_ref_trian)
end

function MappedMeasure(ref_meas::Measure,t::Triangulation)
  MappedMeasure(ref_meas.quad,t)
end

function MappedMeasure(ref_trian::Triangulation,args...;kwargs...)
  ref_quad = CellQuadrature(ref_trian,args...;kwargs...)
  MappedMeasure(ref_quad,t)
end

CellData.get_cell_quadrature(a::MappedMeasure) = a.quad

function CellData.integrate(f,a::MappedMeasure)
  c = reference_integral(f,a.quad,a.cell_map)
  cont = DomainContribution()
  add_contribution!(cont,a.quad.trian,c)
  cont
end

function reference_integral(f::CellField,quad::CellQuadrature,cell_map)
  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """

  @check is_change_possible(trian_f,trian_x) msg

  b = change_domain(f,quad.trian,quad.data_domain_style)
  x = get_cell_points(quad)
  bx = b(x)
  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(IntegrationMap(),bx,quad.cell_weight)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  else
    @notimplemented
  end
end

function reference_integral(a,quad::CellQuadrature,cell_map)
  b = CellField(a,quad.trian,quad.data_domain_style,cell_map)
  reference_integral(b,quad)
end

# function get_mapped_triangulation(t::Triangulation)
#   @abstractmethod
# end

# function get_mapped_triangulation(t::BodyFittedTriangulation)
#   model = get_background_model(t)
#   grid = get_mapped_grid(t)
#   BodyFittedTriangulation(model,grid,t.tface_to_mface)
# end

# function get_mapped_triangulation(t::Geometry.BoundaryTriangulation)
#   trian = get_mapped_triangulation(t.trian)
#   Gridap.BoundaryTriangulation(trian,t.glue)
# end

# function get_mapped_triangulation(t::Geometry.AppendedTriangulation)
#   a = get_mapped_triangulation(t.a)
#   b = get_mapped_triangulation(t.b)
#   lazy_append(a,b)
# end

# function get_mapped_triangulation(t::Union{Interfaces.SubCellTriangulation,ParamSubCellTriangulation})
#   model = get_background_model(t)
#   subcells = get_mapped_subcells(t)
#   Interfaces.SubCellTriangulation(subcells,model)
# end

# function get_mapped_triangulation(t::Union{Interfaces.SubFacetTriangulation,ParamSubFacetTriangulation})
#   model = get_background_model(t)
#   facets = get_mapped_subfacets(t)
#   Interfaces.SubFacetTriangulation(facets,model)
# end

# function get_mapped_grid(t::Triangulation)
#   model = get_background_model(t)
#   grid = get_grid(t)
#   get_mapped_grid(model,grid)
# end

# function get_mapped_grid(model::DiscreteModel,grid::Grid)
#   grid
# end

# function get_mapped_grid(model::DiscreteModel,grid::Geometry.GridPortion)
#   Geometry.GridPortion(get_mapped_grid(model,grid.parent),grid.cell_to_parent_cell)
# end

# function get_mapped_grid(model::DiscreteModel,grid::Geometry.GridView)
#   Geometry.GridView(get_mapped_grid(model,grid.parent),grid.cell_to_parent_cell)
# end

# function get_mapped_grid(model::DiscreteModel,grid::ParamGrid)
#   @abstractmethod
# end

# function get_mapped_grid(model::DiscreteModel,grid::ParamMappedGrid)
#   grid.grid
# end

# function get_mapped_grid(model::DiscreteModel,grid::ParamUnstructuredGrid)
#   node_coordinates = collect1d(get_node_coordinates(model))
#   UnstructuredGrid(
#     node_coordinates,
#     g.cell_node_ids,
#     g.reffes,
#     g.cell_type,
#     g.orientation_style,
#     g.facet_normal
#   )
# end

# function get_mapped_subcells(t::Union{Interfaces.SubCellTriangulation,ParamSubCellTriangulation})
#   model = get_background_model(t)
#   get_mapped_subcells(model,t.subcells)
# end

# function get_mapped_subcells(model::DiscreteModel,subcells::Union{Interfaces.SubCellData,ParamSubCellData})
#   point_to_coords = collect1d(get_node_coordinates(model))
#   Interfaces.SubCellData(
#     subcells.cell_to_points,
#     subcells.cell_to_bgcell,
#     point_to_coords,
#     subcells.point_to_rcoords)
# end

# function get_mapped_subfacets(t::Union{Interfaces.SubFacetTriangulation,ParamSubFacetTriangulation})
#   model = get_background_model(t)
#   get_mapped_subfacets(model,t.subfacets)
# end

# function get_mapped_subfacets(model::DiscreteModel,subfacets::Union{Interfaces.SubFacetData,ParamSubFacetData})
#   point_to_coords = collect1d(get_node_coordinates(model))
#   Interfaces.SubFacetData(
#     subfacets.facet_to_points,
#     subfacets.facet_to_normal,
#     subfacets.facet_to_bgcell,
#     point_to_coords,
#     subfacets.point_to_rcoords)
# end

function mapped_gradient(a::CellField,trian::Triangulation)
  @assert DomainStyle(a) == ReferenceDomain()
  cell_map = get_cell_map(trian)
  cell_∇a = lazy_map(Broadcasting(∇),get_data(a))
  g = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
  similar_cell_field(a,g,get_triangulation(a),DomainStyle(a))
end

const ∇ᵩ = mapped_gradient

function get_mapped_facet_normal(ref_n::CellField,ref_Γ::Triangulation,trian::Triangulation)
  @assert DomainStyle(ref_n) == ReferenceDomain()
  D = num_cell_dims(trian)
  cell_map = get_cell_map(trian)
  facet_map = get_glue(ref_Γ,Val{D}()).tface_to_mface_map
  facet_to_cell = _get_face_to_cell(trian,ref_Γ)
  ref_xΓ = get_data(get_cell_points(ref_Γ))
  ref_x = lazy_map(evaluate,facet_map,ref_xΓ)
  ref_facet_n = get_data(ref_n)
  _get_mapped_facet_normal(cell_map,ref_facet_n,ref_x,ref_xΓ,facet_to_cell)
end

function get_mapped_facet_normal(ref_n::CellField,ref_dΓ::Measure,trian::Triangulation)
  @assert DomainStyle(ref_n) == ReferenceDomain()
  D = num_cell_dims(trian)
  cell_map = get_cell_map(trian)
  facet_map = get_glue(ref_dΓ.quad.trian,Val{D}()).tface_to_mface_map
  facet_to_cell = _get_face_to_cell(trian,ref_dΓ.quad.trian)
  ref_xΓ = get_data(get_cell_points(ref_dΓ.quad))
  ref_x = lazy_map(evaluate,facet_map,ref_xΓ)
  ref_facet_n = get_data(ref_n)
  _get_mapped_facet_normal(cell_map,ref_facet_n,ref_x,ref_xΓ,facet_to_cell)
end

function get_mapped_facet_normal(ref_n::CellField,trian::Triangulation)
  ref_Γ = get_triangulation(ref_n)
  get_mapped_facet_normal(ref_n,ref_Γ,trian)
end

function get_mapped_normal_vector(ref_n::CellField,args...)
  cell_normal = get_mapped_facet_normal(ref_n,args...)
  ref_Γ = get_triangulation(ref_n)
  get_normal_vector(ref_Γ,cell_normal)
end

const nᵩ = get_mapped_normal_vector

# utils

function _get_mapped_cell_Jtx(cell_map,ref_x,facet_to_cell)
  cell_Jt = lazy_map(∇,cell_map)
  facet_Jt = lazy_map(Reindex(cell_Jt),facet_to_cell)
  lazy_map(evaluate,facet_Jt,ref_x)
end

function _get_mapped_facet_normal(cell_map,ref_facet_n,ref_x,ref_xΓ,facet_to_cell)
  cell_Jt = lazy_map(∇,cell_map)
  cell_invJt = lazy_map(Operation(Fields.pinvJt),cell_Jt)
  facet_invJt = lazy_map(Reindex(cell_invJt),facet_to_cell)
  facet_invJtx = lazy_map(evaluate,facet_invJt,ref_x)
  ref_facet_nx = lazy_map(evaluate,ref_facet_n,ref_xΓ)
  facet_nx = lazy_map(Broadcasting(Geometry.push_normal),facet_invJtx,ref_facet_nx)
  lazy_map(constant_field,facet_nx)
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
  @check _model_compatibility(smodel,tmodel)

  sglue = get_glue(strian,Val(Dc))
  tglue = get_glue(ttrian,Val(Dc))
  is_change_possible(sglue,tglue)
end

_get_at_index(a::LazyArray,i) = lazy_param_getindex(a,i)
_get_at_index(a::GenericParamBlock,i) = a.data[i]

function _get_cell_map_on_trian(strian::Triangulation,ttrian::Triangulation)
  scell_map = get_cell_map(strian)
  scell_to_tcell = _get_cell_to_cell(strian,ttrian)
  _get_cell_map_on_trian(scell_map,scell_to_tcell)
end

function _get_cell_map_on_trian(scell_map,scell_to_tcell)
  lazy_map(Reindex(scell_map),scell_to_tcell)
end

function _get_cell_map_on_trian(scell_map::AppendedArray,scell_to_tcell::AppendedArray)
  a = _get_cell_map_on_trian(scell_map.a,scell_to_tcell.a)
  b = _get_cell_map_on_trian(scell_map.b,scell_to_tcell.b)
  lazy_append(a,b)
end

function _get_cell_to_cell(strian::Triangulation,ttrian::Triangulation)
  if strian ≈ ttrian
    IdentityVector(num_cells(strian))
  else
    @assert Utils.is_parent(strian,ttrian) || Utils.isapprox_parent(strian,ttrian)
    _get_cell_to_parent_cell(ttrian)
  end
end

function _get_cell_to_parent_cell(grid::Grid)
  @abstractmethod
end

function _get_cell_to_parent_cell(grid::Geometry.GridView)
  grid.cell_to_parent_cell
end

function _get_cell_to_parent_cell(grid::Geometry.GridPortion)
  grid.cell_to_parent_cell
end

function _get_cell_to_parent_cell(ttrian::Triangulation)
  _get_cell_to_parent_cell(get_grid(ttrian))
end

function _get_cell_to_parent_cell(ttrian::Geometry.TriangulationView)
  ttrian.cell_to_parent_cell
end

function _get_cell_to_parent_cell(ttrian::Geometry.AppendedTriangulation)
  a = _get_cell_to_parent_cell(ttrian.a)
  b = _get_cell_to_parent_cell(ttrian.b)
  offset = num_cells(get_parent(ttrian.a))
  b_offset = lazy_map(+,b,Fill(offset,length(b)))
  lazy_append(a,b_offset)
end

function _get_face_to_cell(strian::Triangulation,ttrian::Triangulation)
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))

  mface_to_sface = sglue.mface_to_tface
  tface_to_mface = tglue.tface_to_mface

  lazy_map(Reindex(mface_to_sface),tface_to_mface)
end

struct EvalConstantField{T} <: Field
  value::T
end

function Fields.ConstantField(value::Union{AbstractArray,ParamBlock})
  EvalConstantField(value)
end

function Arrays.evaluate(f::EvalConstantField,x::AbstractArray{<:Point})
  @assert size(f.value) == size(x)
  f.value
end

function Arrays.evaluate!(c,f::EvalConstantField,x::AbstractArray{<:Point})
  @assert size(f.value) == size(x)
  f.value
end
