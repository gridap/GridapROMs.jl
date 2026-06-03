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

function ReducedCellField(
  f::CellField;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  trian = get_triangulation(f)
  ds = DomainStyle(typeof(f))
  ReducedCellField(get_data(f),trian,ds,order,reduction)
end

CellData.get_data(f::ReducedCellField) = f.cell_field
CellData.get_triangulation(f::ReducedCellField) = f.trian
CellData.DomainStyle(::Type{ReducedCellField{DS}}) where {DS} = DS()

function CellData.similar_cell_field(f::ReducedCellField,cell_data,trian,ds)
  ReducedCellField(cell_data,trian,ds,f.order,f.reduction)
end

function _param_length_or_nothing(data)
  length(data) == 0 && return nothing
  d0 = first(data)
  applicable(param_length,d0) ? param_length(d0) : nothing
end

function _is_parametric_field(f::CellField)
  data = get_data(f)
  plength = _param_length_or_nothing(data)
  !isnothing(plength) && plength > 1
end

_reduce_field(f::CellField;order,reduction) = ReducedCellField(f;order,reduction)

function _rebuild_operation(f::CellData.OperationCellField,args::Tuple)
  CellData.OperationCellField(f.op,args...)
end

function _partition_parametric(args::Tuple)
  p_args = CellField[]
  np_args = CellField[]
  p_flags = Vector{Bool}(undef,length(args))
  for (i,arg) in enumerate(args)
    is_p = _is_parametric_field(arg)
    p_flags[i] = is_p
    if is_p
      push!(p_args,arg)
    else
      push!(np_args,arg)
    end
  end
  return p_args,np_args,p_flags
end

function _maybe_group_parametric(
  f::CellData.OperationCellField,
  args::Tuple;
  order::Int,
  reduction::Reduction
  )

  node = _rebuild_operation(f,args)
  _is_parametric_field(node) || return node

  p_args,np_args,_ = _partition_parametric(args)
  np = length(p_args)
  nn = length(np_args)

  np == 0 && return node

  # Single argument operation: reduce if parametric.
  if length(args) == 1
    return _reduce_field(node;order,reduction)
  end

  op = f.op.op

  if op === (+)
    if np > 1
      pnode = CellData.OperationCellField(f.op,p_args...)
      pred = _reduce_field(pnode;order,reduction)
      if nn == 0
        return pred
      else
        return CellData.OperationCellField(f.op,pred,np_args...)
      end
    else
      return node
    end
  elseif op === (*)
    # Only reduce pure-parametric products; mixed products can change value
    # types (e.g. scalar*vector) and break interpolation consistency.
    if nn == 0 && np > 1
      return _reduce_field(node;order,reduction)
    else
      return node
    end
  else
    # Keep mixed non-multiplicative nodes untouched (e.g. dot with non-parametric du).
    return node
  end
end

function propagate_reduction(
  f::CellField;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  if f isa ReducedCellField
    return f
  elseif f isa CellData.OperationCellField
    args = map(arg -> propagate_reduction(arg;order,reduction),f.args)
    _maybe_group_parametric(f,args;order,reduction)
  elseif _is_parametric_field(f)
    _reduce_field(f;order,reduction)
  else
    f
  end
end

function Arrays.evaluate!(cache,f::ReducedCellField,x::CellPoint)
  cell_field,x = CellData._to_common_domain(f,x)
  
  data = get_data(cell_field)
  plength = _param_length_or_nothing(data)
  if isnothing(plength) || plength <= 1
    return cell_field(x)
  end
  trian = get_triangulation(cell_field)

  reffe = ReferenceFE(lagrangian,Float64,f.order)
  qspace = QuadratureFESpace(trian,reffe)
  pqspace = parameterise(qspace,plength)
  fqh = interpolate(cell_field,pqspace)

  free_vals = get_free_dof_values(fqh)
  _red_free_vals = reduction(f.reduction,get_all_data(free_vals))
  red_free_vals = ConsecutiveParamArray(_red_free_vals)
  rplength = param_length(red_free_vals)
  rpqspace = parameterise(qspace,rplength)
  rfqh = FEFunction(rpqspace,red_free_vals)

  return rfqh(x)
end

struct ReducedFunction{F<:Function,R<:Reduction} <: Function
  f::F
  order::Int
  reduction::R
end

function ReducedFunction(f::Function;order::Int,reduction=Reduction(1e-5;sketch=:sprn))
  ReducedFunction(f,order,reduction)
end

function ℛ(
  f::Function;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  ReducedFunction(f;order,reduction)
end

function ℛ(
  f::CellField;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  ReducedCellField(f;order,reduction)
end

function CellData.CellField(f::ReducedFunction,trian::Triangulation,domain_style::DomainStyle)
  s = size(get_cell_map(trian))
  data = Fill(GenericField(f.f),s)
  ReducedCellField(data,trian,PhysicalDomain(),f.order,f.reduction)
end

struct ReducedIntegrand{A,R}
  object::A
  order::Int
  reduction::R
end

function ReducedIntegrand(
  object;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  ReducedIntegrand(object,order,reduction)
end

function reduced_integrate(
  f::CellField,
  quad::CellQuadrature;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  trian_f = get_triangulation(f)
  trian_x = get_triangulation(quad)

  msg = """\n
    Your are trying to integrate a CellField using a CellQuadrature defined on incompatible
    triangulations. Verify that either the two objects are defined in the same triangulation
    or that the triangulaiton of the CellField is the background triangulation of the CellQuadrature.
    """
  @check is_change_possible(trian_f,trian_x) msg

  b = change_domain(f,quad.trian,quad.data_domain_style)
  b′ = propagate_reduction(b;order,reduction)
  x = get_cell_points(quad)
  bx = b′(x)

  if quad.data_domain_style == PhysicalDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    lazy_map(IntegrationMap(),bx,quad.cell_weight)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == PhysicalDomain()
    cell_map = get_cell_map(quad.trian)
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  elseif quad.data_domain_style == ReferenceDomain() &&
            quad.integration_domain_style == ReferenceDomain()
    cell_map = Fill(GenericField(identity),length(bx))
    cell_Jt = lazy_map(∇,cell_map)
    cell_Jtx = lazy_map(evaluate,cell_Jt,quad.cell_point)
    lazy_map(IntegrationMap(),bx,quad.cell_weight,cell_Jtx)
  else
    @notimplemented
  end
end

function reduced_integrate(
  a,
  quad::CellQuadrature;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  b = CellField(a,quad.trian,quad.data_domain_style)
  reduced_integrate(b,quad;order,reduction)
end

function CellData.integrate(
  a::ReducedIntegrand,
  quad::CellQuadrature
  )

  reduced_integrate(a.object,quad;order=a.order,reduction=a.reduction)
end

function reduced_integrate(
  a,
  m;
  order::Int,
  reduction::Reduction=Reduction(1e-5;sketch=:sprn)
  )

  integrate(ReducedIntegrand(a;order,reduction),m)
end

function _is_integral_call(ex)
  ex isa Expr || return false
  ex.head == :call || return false
  length(ex.args) == 2 || return false
  ex.args[1] == Symbol("∫")
end

function _rewrite_integrals(ex,order_ex)
  ex isa Expr || return ex

  if ex.head == :call && length(ex.args) == 3 && ex.args[1] == :*
    left = _rewrite_integrals(ex.args[2],order_ex)
    right = _rewrite_integrals(ex.args[3],order_ex)
    if _is_integral_call(left)
      integrand = left.args[2]
      return :(GridapROMs.RBSteady.reduced_integrate($integrand,$right;order=$order_ex))
    end
    return Expr(:call,:*,left,right)
  end

  Expr(ex.head,(_rewrite_integrals(arg,order_ex) for arg in ex.args)...)
end

"""
    @reduce_integrals order expr

Rewrites all occurrences of `∫(integrand)*measure` inside `expr` into
`reduced_integrate(integrand,measure;order=order)`.

This applies reduction after `change_domain` and right before point-wise
evaluation in integration, so grouping is performed on the full integrand
CellField at runtime.
"""
macro reduce_integrals(order,expr)
  _rewrite_integrals(expr,order) |> esc
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
