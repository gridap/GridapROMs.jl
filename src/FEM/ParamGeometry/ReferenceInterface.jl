abstract type ConfigurationStyle end
struct ReferenceConfiguration <: ConfigurationStyle end
struct ConfigurationAtIndex <: ConfigurationStyle
  index::Int
end

struct ConfigurationTriangulation{Dt,Dp,B<:ConfigurationStyle} <: Triangulation{Dt,Dp}
  trian::Triangulation{Dt,Dp}
  style::B
end

function get_configuration_triangulation(style::ConfigurationStyle,trian::Triangulation)
  ConfigurationTriangulation(trian,style)
end

function get_configuration_triangulation(trian::Triangulation)
  ConfigurationTriangulation(trian,ReferenceConfiguration())
end

Geometry.get_background_model(t::ConfigurationTriangulation) = get_background_model(t.trian)
Geometry.get_glue(t::ConfigurationTriangulation,::Val{D}) where D = get_glue(t.trian,Val{D}())
Geometry.get_grid(t::ConfigurationTriangulation) = _get_grid(t.style,get_grid(t.trian),get_background_model(t))
Geometry.get_facet_normal(t::ConfigurationTriangulation) = get_facet_normal(t.trian)

_get_grid(s::ConfigurationStyle,g::Grid,m::DiscreteModel) = g
_get_grid(s::ConfigurationStyle,g::ParamGrid,m::DiscreteModel) = @abstractmethod

function _get_grid(s::ConfigurationStyle,g::Geometry.GridPortion,m::DiscreteModel)
  Geometry.GridPortion(_get_grid(s,g.parent,m),g.cell_to_parent_cell)
end

_get_grid(s::ReferenceConfiguration,g::ParamMappedGrid,m::DiscreteModel) = g.grid

function _get_grid(s::ReferenceConfiguration,g::ParamUnstructuredGrid,m::DiscreteModel)
  node_coordinates = collect1d(get_node_coordinates(m))
  UnstructuredGrid(
    node_coordinates,
    g.cell_node_ids,
    g.reffes,
    g.cell_type,
    g.orientation_style,
    g.facet_normal
  )
end

function _get_grid(s::ConfigurationAtIndex,g::ParamMappedGrid,m::DiscreteModel)
  UnstructuredGrid(
    _get_at_index(g.node_coords,s.index),
    get_cell_node_ids(g),
    get_reffes(g),
    collect1d(get_cell_type(g))
    )
end

function _get_grid(s::ConfigurationAtIndex,g::ParamUnstructuredGrid,m::DiscreteModel)
  UnstructuredGrid(
    _get_at_index(g.node_coordinates,s.index),
    g.cell_node_ids,
    g.reffes,
    g.cell_type,
    g.orientation_style,
    g.facet_normal
  )
end

struct ReferenceMeasure <: Measure
  quad::CellQuadrature
  cell_map
end

function ReferenceMeasure(trian::Triangulation,args...;kwargs...)
  cell_map = get_cell_map(trian)
  ref_trian = get_configuration_triangulation(trian)
  ref_quad = CellQuadrature(ref_trian,args...;kwargs...)
  ReferenceMeasure(ref_quad,cell_map)
end

CellData.get_cell_quadrature(a::ReferenceMeasure) = a.quad

function CellData.integrate(f,a::ReferenceMeasure)
  c = reference_integrate(f,a.quad,a.cell_map)
  cont = DomainContribution()
  add_contribution!(cont,a.quad.trian,c)
  cont
end

function reference_integrate(f::CellField,quad::CellQuadrature,cell_map)
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

function reference_integrate(a,quad::CellQuadrature,cell_map)
  b = CellField(a,quad.trian,quad.data_domain_style,cell_map)
  reference_integrate(b,quad)
end

# utils

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

function ref_gradient(a::CellField,trian::Triangulation)
  @assert DomainStyle(a) == ReferenceDomain()
  cell_∇a = lazy_map(Broadcasting(∇),get_data(a))
  cell_map = get_cell_map(trian)
  g = lazy_map(Broadcasting(push_∇),cell_∇a,cell_map)
  similar_cell_field(a,g,get_triangulation(a),DomainStyle(a))
end

const ∇₀ = ref_gradient

_get_at_index(a::LazyArray,i) = lazy_param_getindex(a,i)
_get_at_index(a::GenericParamBlock,i) = a.data[i]
