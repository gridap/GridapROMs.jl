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

struct QuadCellField <: CellField
  cell_field::AbstractArray
  trian::Triangulation
  function QuadCellField(cell_field::AbstractArray,trian::Triangulation)
    new(Fields.MemoArray(cell_field),trian)
  end
end

CellData.get_data(f::QuadCellField) = f.cell_field
CellData.get_triangulation(f::QuadCellField) = f.trian
CellData.DomainStyle(::Type{QuadCellField}) = PhysicalDomain()

for T in (:ReferenceDomain,:PhysicalDomain)
  @eval begin
    function CellData.change_domain(
      a::QuadCellField,
      strian::Triangulation,
      source::ReferenceDomain,
      ttrian::Triangulation,
      target::$T
      )

      error("Should not be here")
    end
    
    function CellData.change_domain(
      a::QuadCellField,
      strian::Triangulation,
      source::PhysicalDomain,
      ttrian::Triangulation,
      target::$T
      )

      if strian === ttrian
        return QuadCellField(get_data(a),ttrian)
      end

      @check is_change_possible(strian,ttrian)
      D = num_cell_dims(strian)
      sglue = get_glue(strian,Val(D))
      tglue = get_glue(ttrian,Val(D))

      sface_to_field = get_data(a)
      mface_to_sface = sglue.mface_to_tface
      tface_to_mface = tglue.tface_to_mface
      mface_to_field = extend(sface_to_field,mface_to_sface)
      tface_to_field = lazy_map(Reindex(mface_to_field),tface_to_mface)

      GenericCellField(tface_to_field,ttrian,target)
    end
  end
end

# function Arrays.evaluate!(cache,f::QuadCellField,x::CellPoint)
#   @check get_triangulation(f) == get_triangulation(x) 
#   get_data(f)
# end

# function Arrays.evaluate!(cache,k::Operation,a::QuadCellField,b::QuadCellField)
#   _operate_cellfields(k,a...)
# end

function reduce_cell_values(
  fields::Fill{<:ParamBlock{<:Fields.GenericField}},
  trian::Triangulation,
  order::Int;
  tol=1e-5,
  red=Reduction(tol;sketch=:sprn),
  )
  
  reffe = ReferenceFE(lagrangian,Float64,order)
  qspace = QuadratureFESpace(trian,reffe)
  qdofs = get_fe_dof_basis(qspace)
  cell_fields = GenericCellField(fields,trian,DomainStyle(qdofs))
  cell_values = qdofs(cell_fields)

  plength = param_length(fields.value)
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
  
  return QuadCellField(reduced_cell_values,trian)
end

function reduce_cell_values(a::LazyArray{G,T,N,F},args...;kwargs...) where {G,T,N,F}
  if G<:AbstractArray{<:ParamBlock{<:Fields.GenericField}}
    rmaps = reduce_cell_values(a.maps,args...;kwargs...)
    return LazyArray(T,N,rmaps,a.args...)
  else    
    rargs = map(ai->reduce_cell_values(ai,args...;kwargs...),a.args)
    return LazyArray(T,N,a.maps,rargs...)
  end
end

function reduce_integral(a::AbstractArray,args...;kwargs...)
  reduce_cell_values(a,args...;kwargs...)
end

# struct ReduceCellField{DS,T} <: CellField
#   cell_field::GenericCellField{DS}
#   function ReduceCellField(cell_field::GenericCellField{DS}) where DS
#     T = eltype(get_data(cell_field))
#     new{DS,T}(cell_field)
#   end
# end

# function ReduceCellField(f::CellField)
#   @abstractmethod
# end

# function ReduceCellField(f::OperationCellField)
#   OperationCellField(f.op,map(ReduceCellField,f.args)...)
# end

# function ReduceCellField(f::CellFieldAt{T}) where T
#   CellFieldAt{T}(ReduceCellField(f.parent))
# end

# function ReduceCellField(f::SkeletonCellFieldPair)
#   cf_plus = ReduceCellField(f.cf_plus)
#   cf_minus = ReduceCellField(f.cf_minus)
#   SkeletonCellFieldPair(cf_plus,cf_minus,f.trian)
# end

# CellData.get_data(f::ReduceCellField) = get_data(f.cell_field)
# CellData.get_triangulation(f::ReduceCellField) = get_triangulation(f.cell_field)
# CellData.DomainStyle(::Type{<:ReduceCellField{DS}}) where DS = DS()

# function reduced_field(cache,f::CellField,x::CellPoint)
#   _f,_x = CellData._to_common_domain(f,x)
#   cell_field = get_data(_f)
#   cell_point = get_data(_x)
#   lazy_map(evaluate,cell_field,cell_point)
# end

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
  @check _compatible_quad_nodes(quad.cell_point) "Quadrature with incompatible nodes, got $(quad.cell_point)."
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

# function CellData._to_common_domain(a::QuadCellField)
#   a 
# end

# for T in (:CellField,:QuadCellField)
#   @eval begin
#     function CellData._to_common_domain(a::QuadCellField,b::$T)
#       trian_a = get_triangulation(a)
#       trian_b = get_triangulation(b)
#       sa_tb = is_change_possible(trian_a,trian_b)
#       sb_ta = is_change_possible(trian_b,trian_a)
#       @check sa_tb || sb_ta "Cannot find common domain for $(typeof(a)) and $(typeof(b))."
#       return a,change_domain(b,trian_a,DomainStyle(a))
#     end
#   end
# end

# function CellData._to_common_domain(a::CellField,b::QuadCellField)
#   reverse(CellData._to_common_domain(b,a))
# end