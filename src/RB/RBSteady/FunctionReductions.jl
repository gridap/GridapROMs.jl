function QuadratureReferenceFE(
  ::Type{T},
  p::Polytope{D},
  quad::CellQuadrature,
  orders;
  space::Symbol=ReferenceFEs._default_space(p),
  poly_type=Monomial
  ) where {T,D}

  quad_nodes = _get_quad_nodes(quad)
  if space == :P && is_n_cube(p)
    @notimplemented 
    return ReferenceFEs._PDiscRefFE(T,p,orders,poly_type)
  elseif space == :S && is_n_cube(p)
    @notimplemented
    SerendipityRefFE(T,p,orders;poly_type)
  else
    if any(map(i->i==0,orders)) && !all(map(i->i==0,orders))
      @check poly_type == Monomial "Continuous-Discontinuous element only implemented using Monomial pre-bases,got $poly_type."
      cont = map(i -> i == 0 ? DISC : CONT,orders)
      return _cd_quad_lagrangian_ref_fe(T,p,quad_nodes,orders,cont)
    else
      return _quad_lagrangian_ref_fe(T,p,quad_nodes,orders,poly_type)
    end
  end
end

function QuadratureReferenceFE(
  ::Type{T},
  p::Polytope{D},
  quad::CellQuadrature,
  order::Int;
  space::Symbol=ReferenceFEs._default_space(p),
  poly_type=Monomial
  ) where {T,D}

  orders = tfill(order,Val{D}())
  QuadratureReferenceFE(T,p,quad,orders;space,poly_type)
end

function QuadratureReferenceFE(
  polytope::Polytope,
  quad::CellQuadrature,
  ::Lagrangian,
  ::Type{T},
  orders::Union{Integer,Tuple{Vararg{Integer}}};
  kwargs...) where T

  QuadratureReferenceFE(T,polytope,quad,orders;kwargs...)
end

function QuadratureReferenceFE(quad::CellQuadrature,args...;kwargs...)
  trian = get_triangulation(quad)
  ctype_to_polytope = get_polytopes(trian)
  cell_to_ctype = get_cell_type(trian)
  ctype_to_reffe = map(p->QuadratureReferenceFE(p,quad,args...;kwargs...),ctype_to_polytope)
  cell_to_reffe = expand_cell_data(ctype_to_reffe,cell_to_ctype)
  return cell_to_reffe
end

function QuadratureFESpace(
  trian::Triangulation,
  reffe::Union{ReferenceFE,Tuple{<:Union{ReferenceFEName,Symbol},Any,Any}};
  kwargs...
  )

  reffe_name,reffe_args,reffe_kwargs = reffe
  degree = _get_degree(reffe_args...)
  quad = CellQuadrature(trian,degree)
  cell_reffe = QuadratureReferenceFE(quad,reffe_name,reffe_args...;reffe_kwargs...)
  FESpace(trian,cell_reffe;conformity=:L2,kwargs...)
end

struct ReducedCellField{DS<:DomainStyle} <: CellField
  cell_field::AbstractArray
  trian::Triangulation
  domain_style::DS
  order::Int
  reduction::Reduction

  function ReducedCellField(
    cell_field::AbstractArray,
    trian::Triangulation,
    domain_style::DomainStyle=PhysicalDomain(),
    order::Int=2,
    reduction::Reduction=Reduction(1e-5;sketch=:sprn)
    )

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style,order,reduction)
  end
end

CellData.get_data(f::ReducedCellField) = f.cell_field
CellData.get_triangulation(f::ReducedCellField) = f.trian
CellData.DomainStyle(::Type{ReducedCellField{DS}}) where {DS} = DS()

function CellData.similar_cell_field(f::ReducedCellField,cell_data,trian,ds)
  ReducedCellField(cell_data,trian,ds,f.order,f.reduction)
end

function Arrays.evaluate!(cache,f::ReducedCellField,x::CellPoint)
  cell_field, = CellData._to_common_domain(f,x)
  
  data = get_data(cell_field)
  trian = get_triangulation(cell_field)

  reffe = ReferenceFE(lagrangian,Float64,f.order)
  qspace = QuadratureFESpace(trian,reffe)
  qdofs = get_fe_dof_basis(qspace)
  cell_values = qdofs(cell_field)

  plength = param_length(first(data))
  free_values = parameterise(zero_free_values(qspace),plength)
  diri_values = parameterise(zero_dirichlet_values(qspace),plength)
  gather_free_and_dirichlet_values!(free_values,diri_values,qspace,cell_values)

  _reduced_free_values = reduction(f.reduction,get_all_data(free_values))
  reduced_free_values = ConsecutiveParamArray(_reduced_free_values)
  rplength = param_length(reduced_free_values)
  reduced_diri_values = parameterise(zero_dirichlet_values(qspace),rplength)
  scatter_free_and_dirichlet_values(
    qspace,
    reduced_free_values,
    reduced_diri_values
  )
end

struct ReducedFunction{F<:Function,R<:Reduction} <: Function
  f::F
  order::Int
  reduction::R
end

function ReducedFunction(f::Function;order::Int,reduction=Reduction(1e-5;sketch=:sprn))
  ReducedFunction(f,order,reduction)
end

const ℛ = ReducedFunction

function CellData.CellField(f::ReducedFunction,trian::Triangulation,domain_style::DomainStyle)
  s = size(get_cell_map(trian))
  data = Fill(GenericField(f.f),s)
  ReducedCellField(data,trian,PhysicalDomain(),f.order,f.reduction)
end

# utils 

function _quad_lagrangian_ref_fe(
  ::Type{T},
  p::ExtrusionPolytope{D},
  nodes,
  orders,
  poly_type
  ) where {T,D}

  basis = ReferenceFEs.compute_poly_basis(T,p,orders,poly_type)
  _,face_own_nodes = compute_nodes(p,orders)
  dofs = LagrangianDofBasis(T,nodes)
  reffaces = compute_lagrangian_reffaces(T,p,orders)

  nnodes = length(dofs.nodes)
  ndofs = length(dofs.dof_to_node)
  metadata = reffaces
  _reffaces = vcat(reffaces...)
  face_nodes = ReferenceFEs._generate_face_nodes(nnodes,face_own_nodes,p,_reffaces)
  face_own_dofs = ReferenceFEs._generate_face_own_dofs(face_own_nodes,dofs.node_and_comp_to_dof)

  if all(map(i->i==0,orders) ) && D>0
    conf = L2Conformity()
  else
    conf = GradConformity()
  end

  reffe = GenericRefFE{typeof(conf)}(
    ndofs,
    p,
    basis,
    dofs,
    conf,
    metadata,
    face_own_dofs)
  GenericLagrangianRefFE(reffe,face_nodes)
end

function _cd_quad_lagrangian_ref_fe(
  ::Type{T},
  p::ExtrusionPolytope{D},
  nodes,
  orders,
  cont
  ) where {T,D}

  @check isa(p,ExtrusionPolytope)

  prebasis = compute_monomial_basis(T,p,orders)

  _,face_own_nodes = ReferenceFEs.cd_compute_nodes(p,orders)
  dofs = LagrangianDofBasis(T,nodes)

  ndofs = length(dofs.dof_to_node)

  face_own_nodes = ReferenceFEs._compute_cd_face_own_nodes(p,orders,cont)
  face_nodes = ReferenceFEs._compute_face_nodes(p,face_own_nodes)

  face_own_dofs = ReferenceFEs._generate_face_own_dofs(face_own_nodes,dofs.node_and_comp_to_dof)

  data = nothing

  conf = CDConformity(Tuple(cont))

  reffe = GenericRefFE{typeof(conf)}(
    ndofs,
    p,
    prebasis,
    dofs,
    conf,
    data,
    face_own_dofs
  )

  GenericLagrangianRefFE(reffe,face_nodes)
end

_get_degree(args...) = @abstractmethod
_get_degree(::Type{T},order::Integer;γ=2) where T = γ*order
_get_degree(::Type{T},orders::Tuple{Vararg{Integer}};γ=2) where T = γ*maximum(orders)  

function _get_quad_nodes(quad::CellQuadrature)
  @check _compatible_quad_nodes(quad.cell_point) "Quadrature with incompatible nodes,got $(quad.cell_point)."
  first(quad.cell_point)
end

function _compatible_quad_nodes(cell_point)
  @abstractmethod 
end

function _compatible_quad_nodes(cell_point::Fill{<:AbstractVector{<:Point}})
  true 
end

function _compatible_quad_nodes(cell_point::AbstractArray{<:AbstractVector{<:Point}})
  p1 = first(cell_point)
  all([p == p1 for p in cell_point]) 
end
