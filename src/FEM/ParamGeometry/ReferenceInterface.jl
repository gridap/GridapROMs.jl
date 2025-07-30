struct ReferenceMeasure <: Measure
  quad::CellQuadrature
  cell_map
end

function ReferenceMeasure(t::Triangulation,args...;kwargs...)
  cell_map = get_cell_map(t)
  ref_trian = get_reference_triangulation(t)
  ref_quad = CellQuadrature(ref_trian,args...;kwargs...)
  ReferenceMeasure(ref_quad,cell_map)
end

CellData.get_cell_quadrature(a::ReferenceMeasure) = a.quad

function CellData.integrate(f,a::ReferenceMeasure)
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

function get_reference_triangulation(t::Triangulation)
  @abstractmethod
end

function get_reference_triangulation(t::BodyFittedTriangulation)
  model = get_background_model(t)
  grid = get_reference_grid(t)
  BodyFittedTriangulation(model,grid,t.tface_to_mface)
end

function get_reference_triangulation(t::Geometry.BoundaryTriangulation)
  trian = get_reference_triangulation(t.trian)
  Gridap.BoundaryTriangulation(trian,t.glue)
end

function get_reference_triangulation(t::Geometry.AppendedTriangulation)
  a = get_reference_triangulation(t.a)
  b = get_reference_triangulation(t.b)
  lazy_append(a,b)
end

function get_reference_triangulation(t::Union{Interfaces.SubCellTriangulation,ParamSubCellTriangulation})
  model = get_background_model(t)
  subcells = get_reference_subcells(t)
  Interfaces.SubCellTriangulation(subcells,model)
end

function get_reference_triangulation(t::Union{Interfaces.SubFacetTriangulation,ParamSubFacetTriangulation})
  model = get_background_model(t)
  facets = get_reference_subfacets(t)
  Interfaces.SubFacetTriangulation(facets,model)
end

function get_reference_grid(t::Triangulation)
  model = get_background_model(t)
  grid = get_grid(t)
  get_reference_grid(model,grid)
end

function get_reference_grid(model::DiscreteModel,grid::Grid)
  grid
end

function get_reference_grid(model::DiscreteModel,grid::Geometry.GridPortion)
  Geometry.GridPortion(get_reference_grid(model,grid.parent),grid.cell_to_parent_cell)
end

function get_reference_grid(model::DiscreteModel,grid::Geometry.GridView)
  Geometry.GridView(get_reference_grid(model,grid.parent),grid.cell_to_parent_cell)
end

function get_reference_grid(model::DiscreteModel,grid::ParamGrid)
  @abstractmethod
end

function get_reference_grid(model::DiscreteModel,grid::ParamMappedGrid)
  grid.grid
end

function get_reference_grid(model::DiscreteModel,grid::ParamUnstructuredGrid)
  node_coordinates = collect1d(get_node_coordinates(model))
  UnstructuredGrid(
    node_coordinates,
    g.cell_node_ids,
    g.reffes,
    g.cell_type,
    g.orientation_style,
    g.facet_normal
  )
end

function get_reference_subcells(t::Union{Interfaces.SubCellTriangulation,ParamSubCellTriangulation})
  model = get_background_model(t)
  get_reference_subcells(model,t.subcells)
end

function get_reference_subcells(model::DiscreteModel,subcells::Union{Interfaces.SubCellData,ParamSubCellData})
  point_to_coords = collect1d(get_node_coordinates(model))
  Interfaces.SubCellData(
    subcells.cell_to_points,
    subcells.cell_to_bgcell,
    point_to_coords,
    subcells.point_to_rcoords)
end

function get_reference_subfacets(t::Union{Interfaces.SubFacetTriangulation,ParamSubFacetTriangulation})
  model = get_background_model(t)
  get_reference_subfacets(model,t.subfacets)
end

function get_reference_subfacets(model::DiscreteModel,subfacets::Union{Interfaces.SubFacetData,ParamSubFacetData})
  point_to_coords = collect1d(get_node_coordinates(model))
  Interfaces.SubFacetData(
    subfacets.facet_to_points,
    subfacets.facet_to_normal,
    subfacets.facet_to_bgcell,
    point_to_coords,
    subfacets.point_to_rcoords)
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
