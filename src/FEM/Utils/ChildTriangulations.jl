struct ChildTriangulation{Dc,Dp,A,B} <: Triangulation{Dc,Dp}
  parent::A
  cell_to_parent_cell::B
  function ChildTriangulation(parent::Triangulation,cell_to_parent_cell::AbstractArray)
    Dc = num_cell_dims(parent)
    Dp = num_point_dims(parent)
    A = typeof(parent)
    B = typeof(cell_to_parent_cell)
    new{Dc,Dp,A,B}(parent,cell_to_parent_cell)
  end
end

function ChildTriangulation(parent::Triangulation,parent_cell_to_mask::AbstractArray{Bool})
  cell_to_parent_cell = findall(collect1d(parent_cell_to_mask))
  ChildTriangulation(parent,cell_to_parent_cell)
end

function ChildTriangulation(parent::Triangulation,parent_cell_to_mask::AbstractVector{Bool})
  cell_to_parent_cell = findall(parent_cell_to_mask)
  ChildTriangulation(parent,cell_to_parent_cell)
end

function Geometry.get_background_model(trian::ChildTriangulation)
  get_background_model(trian.parent)
end

function Geometry.get_grid(trian::ChildTriangulation)
  view(get_grid(trian.parent),trian.cell_to_parent_cell)
end

function Geometry.get_glue(t::ChildTriangulation,::Val{d}) where d
  parent = get_glue(t.parent,Val(d))
  parent === nothing && return nothing
  view(parent,t.cell_to_parent_cell)
end

function Geometry.get_facet_normal(trian::ChildTriangulation)
  restrict(get_facet_normal(trian.parent),trian.cell_to_parent_cell)
end

function Geometry.get_cell_map(trian::ChildTriangulation)
  restrict(get_cell_map(trian.parent),trian.cell_to_parent_cell)
end

struct ChildCellQuadrature{DDS,IDS} <: CellDatum
  parent::CellQuadrature{DDS,IDS}
  child_trian::ChildTriangulation
end

function CellData.CellQuadrature(
  trian::ChildTriangulation,
  degree::Integer;
  data_domain_style::DomainStyle=ReferenceDomain(),
  integration_domain_style::DomainStyle=PhysicalDomain(),
  kwargs...
  )

  parent = CellQuadrature(
    trian.parent, 
    degree;
    data_domain_style=data_domain_style,
    integration_domain_style=integration_domain_style,
    kwargs...
  )
  ChildCellQuadrature(parent,trian)
end

CellData.get_data(f::ChildCellQuadrature) = lazy_map(Reindex(get_data(f.parent)),f.child_trian.cell_to_parent_cell)
CellData.get_triangulation(f::ChildCellQuadrature) = f.child_trian
CellData.DomainStyle(::Type{<:ChildCellQuadrature{DDS,IDS}}) where {DDS,IDS} = DDS()

function CellData.change_domain(a::ChildCellQuadrature,::ReferenceDomain,::PhysicalDomain)
  @notimplemented
end

function CellData.change_domain(a::ChildCellQuadrature,::PhysicalDomain,::ReferenceDomain)
  @notimplemented
end

function CellData.get_cell_points(a::ChildCellQuadrature)
  @notimplemented
end

function CellData.integrate(f::CellField,quad::ChildCellQuadrature)
  fq = integrate(f,quad.parent)
  lazy_map(Reindex(fq),quad.child_trian.cell_to_parent_cell)
end

function CellData.integrate(a,quad::ChildCellQuadrature)
  aq = integrate(a,quad.parent)
  lazy_map(Reindex(aq),quad.child_trian.cell_to_parent_cell)
end

Base.:*(a::Integrand,b::ChildCellQuadrature) = integrate(a.object,b)
Base.:*(b::ChildCellQuadrature,a::Integrand) = integrate(a.object,b)

struct ChildMeasure <: Measure
  quad::ChildCellQuadrature
end

CellData.Measure(q::ChildCellQuadrature) = ChildMeasure(q)

function CellData.Measure(trian::ChildTriangulation,args...;kwargs...)
  quad = CellQuadrature(trian,args...;kwargs...)
  ChildMeasure(quad)
end

CellData.get_cell_quadrature(a::ChildMeasure) = a.quad

function CellData.integrate(f,b::ChildMeasure)
  c = integrate(f,b.quad)
  cont = DomainContribution()
  add_contribution!(cont,b.quad.child_trian,c)
  cont
end

# relationships 

function is_parent(parent::Triangulation,child::Triangulation)
  false 
end

function is_parent(parent::Triangulation,child::ChildTriangulation)
  parent === child.parent
end

function find_trian_permutation(a,b,cmp::Function)
  map(a -> findfirst(b -> cmp(a,b),b),a)
end

function find_trian_permutation(a,b)
  cmp(a,b) = a === b || is_parent(a,b)
  find_trian_permutation(a,b,cmp)
end

"""
    order_domains(
      parents::Tuple{Vararg{Triangulation}},
      children::Tuple{Vararg{Triangulation}}
      ) -> Tuple{Vararg{Triangulation}}

Orders the triangulation children in the same way as the triangulation parents
"""
function order_domains(parents,children)
  @check length(parents) == length(children)
  perm = find_trian_permutation(parents,children)
  map(p->children[p],perm)
end

function change_triangulation(old::Tuple,new::Tuple)
  perm = find_trian_permutation(old,new)
  new′ = ()
  for p in perm
    new′ = (new′...,new[p])
  end
  return new′
end

function change_triangulation(old::Tuple,new::AbstractArray{<:Tuple})
  map(n -> change_triangulation(old,n),new)
end

function change_triangulation(old::AbstractArray{<:Tuple},new::Tuple)
  map(o -> change_triangulation(o,new),old)
end

function change_triangulation(old::AbstractArray{<:Tuple},new::AbstractArray{<:Tuple})
  map(change_triangulation,old,new)
end