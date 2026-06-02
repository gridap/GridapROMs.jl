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

  function ReducedCellField(
    cell_field::AbstractArray,
    trian::Triangulation,
    domain_style::DomainStyle=PhysicalDomain(),
    )

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style)
  end
end

function ReducedCellField(f::CellField;kwargs...)
  @abstractmethod
end

function ReducedCellField(
  cell_field::GenericCellField;
  order::Int,
  tol=1e-5,
  red=Reduction(tol;sketch=:sprn)
  ) 

  data = get_data(cell_field)
  trian = get_triangulation(cell_field)

  reffe = ReferenceFE(lagrangian,Float64,order)
  qspace = QuadratureFESpace(trian,reffe)
  qdofs = get_fe_dof_basis(qspace)
  cell_values = qdofs(cell_field)

  plength = param_length(first(data))
  free_values = parameterise(zero_free_values(qspace),plength)
  diri_values = parameterise(zero_dirichlet_values(qspace),plength)
  gather_free_and_dirichlet_values!(free_values,diri_values,qspace,cell_values)

  _reduced_free_values = reduction(red,get_all_data(free_values))
  reduced_free_values = ConsecutiveParamArray(_reduced_free_values)
  rplength = param_length(reduced_free_values)
  reduced_diri_values = parameterise(zero_dirichlet_values(qspace),rplength)
  reduced_cell_values = scatter_free_and_dirichlet_values(
    qspace,
    reduced_free_values,
    reduced_diri_values
  )

  return ReducedCellField(reduced_cell_values,trian,PhysicalDomain())
end

function ReducedCellField(f::CellData.OperationCellField;kwargs...)
  args = map(a -> ReducedCellField(a;kwargs...),f.args)
  CellData.OperationCellField(f.op,args...)
end

function ReducedCellField(f::CellData.CellFieldAt{T};kwargs...) where T
  CellData.CellFieldAt{T}(ReducedCellField(f.parent;kwargs...))
end

function ReducedCellField(f::SkeletonCellFieldPair;kwargs...)
  cf_plus = ReducedCellField(f.cf_plus;kwargs...)
  cf_minus = ReducedCellField(f.cf_minus;kwargs...)
  SkeletonCellFieldPair(cf_plus,cf_minus,f.trian)
end

CellData.get_data(f::ReducedCellField) = f.cell_field
CellData.get_triangulation(f::ReducedCellField) = f.trian
CellData.DomainStyle(::Type{ReducedCellField{DS}}) where {DS} = DS()

function CellData.change_domain(a::ReducedCellField,::PhysicalDomain,::PhysicalDomain)
  a
end

function CellData.change_domain(a::ReducedCellField,::ReferenceDomain,::ReferenceDomain)
  a
end

function CellData.change_domain(a::ReducedCellField,::PhysicalDomain,::ReferenceDomain)
  ReducedCellField(get_data(a),get_triangulation(a),ReferenceDomain())
end

function CellData.change_domain(a::ReducedCellField,::ReferenceDomain,::PhysicalDomain)
  ReducedCellField(get_data(a),get_triangulation(a),PhysicalDomain())
end

function CellData.change_domain(
  a::ReducedCellField,
  strian::Triangulation,
  ::PhysicalDomain,
  ttrian::Triangulation,
  ::PhysicalDomain,
  )

  data = _change_red_cell_data(get_data(a),strian,ttrian)
  ReducedCellField(data,ttrian,PhysicalDomain())
end

function CellData.change_domain(
  a::ReducedCellField,
  strian::Triangulation,
  ::PhysicalDomain,
  ttrian::Triangulation,
  ::ReferenceDomain,
  )

  data = _change_red_cell_data(get_data(a),strian,ttrian)
  ReducedCellField(data,ttrian,ReferenceDomain())
end

function CellData.change_domain(
  a::ReducedCellField,
  strian::Triangulation,
  ::ReferenceDomain,
  ttrian::Triangulation,
  ::PhysicalDomain,
  )

  data = _change_red_cell_data(get_data(a),strian,ttrian)
  ReducedCellField(data,ttrian,PhysicalDomain())
end

function CellData.change_domain(
  a::ReducedCellField,
  strian::Triangulation,
  ::ReferenceDomain,
  ttrian::Triangulation,
  ::ReferenceDomain,
  )

  data = _change_red_cell_data(get_data(a),strian,ttrian)
  ReducedCellField(data,ttrian,ReferenceDomain())
end

function Arrays.evaluate!(cache,f::ReducedCellField,x::CellPoint)
  trian_x = get_triangulation(x)
  domain_x = DomainStyle(x)
  fx = change_domain(f,trian_x,domain_x)
  get_data(fx)
end

struct ReducedFunction{F<:Function} <: Function
  f::F
end

const ℛ = ReducedFunction

function ReducedCellField(f::ReducedFunction,trian::Triangulation,domain_style::DomainStyle;kwargs...)
  s = size(get_cell_map(trian))
  data = Fill(GenericField(f.f),s)
  cell_field = GenericCellField(data,trian,PhysicalDomain())
  ReducedCellField(cell_field;kwargs...)
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
  all([p .== p1 for p in cell_point]) 
end

function _change_red_cell_data(
  data::AbstractArray,
  strian::Triangulation,
  ttrian::Triangulation,
  )

  if strian === ttrian
    return data
  end

  @check is_change_possible(strian,ttrian)
  D = num_cell_dims(strian)
  sglue = get_glue(strian,Val(D))
  tglue = get_glue(ttrian,Val(D))

  mface_to_field = extend(data,sglue.mface_to_tface)
  lazy_map(Reindex(mface_to_field),tglue.tface_to_mface)
end

function CellData._convert_to_cellfields(_a::ReducedFunction,b::CellField)
  target_domain = DomainStyle(b)
  target_trian = get_triangulation(b)
  order = get_polynomial_order(b)
  a = ReducedCellField(_a,target_trian,target_domain;order)
  return (a,b)
end

function CellData._convert_to_cellfields(a::CellField,b::ReducedFunction)
  reverse(CellData._convert_to_cellfields(b,a)...)
end