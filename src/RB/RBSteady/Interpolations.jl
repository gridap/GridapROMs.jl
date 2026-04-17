abstract type Interpolation end

Interpolation(args...) = @abstractmethod

get_integration_cells(a::Interpolation,args...) = Int32[]
get_cell_idofs(a::Interpolation) = empty_table(Int,Int32,0)
get_owned_icells(a::Interpolation,args...) = Int[]
move_interpolation(a::Interpolation,args...) = a

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,x::Any)
  cache
end

function reduced_triangulation(trian::Triangulation,a::Interpolation)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end

# Empty interpolation

struct EmptyInterpolation <: Interpolation end

function Interpolation(red::HyperReduction)
  EmptyInterpolation()
end

get_integration_cells(a::EmptyInterpolation,trian::AppendedTriangulation) = lazy_append(Int32[],Int32[])

# EIM interpolation

struct GreedyInterpolation{A,B<:IntegrationDomain} <: Interpolation
  interpolation::A
  domain::B
end

for (T,f) in zip((:MDEIMHyperReduction,:SOPTHyperReduction),(:empirical_interpolation,:s_opt))
  @eval begin
    function Interpolation(
      red::$T,
      basis::Projection,
      trian::Triangulation,
      test::RBSpace
      )

      rows,interp = $f(basis)
      factor = lu(interp)
      domain = IntegrationDomain(trian,test,rows)
      GreedyInterpolation(factor,domain)
    end

    function Interpolation(
      red::$T,
      basis::Projection,
      trian::Triangulation,
      trial::RBSpace,
      test::RBSpace
      )

      (rows,cols),interp = $f(basis)
      factor = lu(interp)
      domain = IntegrationDomain(trian,trial,test,rows,cols)
      GreedyInterpolation(factor,domain)
    end
  end
end

get_integration_cells(a::GreedyInterpolation,args...) = get_integration_cells(a.domain,args...)
get_cell_idofs(a::GreedyInterpolation) = get_cell_idofs(a.domain)
get_owned_icells(a::GreedyInterpolation,args...) = get_owned_icells(a,get_integration_cells(a,args...))
get_owned_icells(a::GreedyInterpolation,cells::AbstractVector) = get_owned_icells(a.domain,cells)

function FESpaces.interpolate!(cache::AbstractArray,a::GreedyInterpolation,b::AbstractArray)
  ldiv!(cache,a.interpolation,b)
  cache
end

function FESpaces.interpolate!(cache::AbstractArray,a::GreedyInterpolation,r::AbstractRealisation)
  interpolate!(cache,a.interpolation,r)
  cache
end

function move_interpolation(a::GreedyInterpolation,args...)
  domain = move_integration_domain(a.domain,args...)
  GreedyInterpolation(a.interpolation,domain)
end

# RBF interpolation

struct RBFInterpolation{A<:Interpolator} <: Interpolation
  interpolation::A
end

function Interpolation(red::RBFHyperReduction,a::Projection,s::Snapshots)
  strategy = interp_strategy(red)
  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realisation(s)
  red_data = _get_at_domain(s,inds)
  coeff = allocate_coefficient(a,r)
  ldiv!(coeff,factor,red_data)
  interp = Interpolator(r,coeff,strategy)
  RBFInterpolation(interp)
end

get_integration_cells(a::RBFHyperReduction,trian::AppendedTriangulation) = lazy_append(Int32[],Int32[])

function FESpaces.interpolate!(cache::AbstractArray,a::RBFInterpolation,b::AbstractArray)
  interpolate!(cache,a.interpolation,b)
  cache
end

function FESpaces.interpolate!(cache::AbstractArray,a::RBFInterpolation,r::AbstractRealisation)
  interpolate!(cache,a.interpolation,r)
  cache
end

function RadialBasisFunctions.Interpolator(
  x::Realisation,
  y::ConsecutiveParamArray,
  basis::AbstractRadialBasis=PHS()
  ) 

  dim = length(first(x))
  k = param_length(x)
  npoly = binomial(dim+basis.poly_deg,basis.poly_deg)
  n = k + npoly
  mon = MonomialBasis(dim,basis.poly_deg)
  data_type = promote_type(eltype(first(x.params)),eltype2(y))
  A = Symmetric(zeros(data_type,n,n))
  RadialBasisFunctions._build_collocation_matrix!(A,x.params,basis,mon,k)
  l = innerlength(y)
  b = zeros(data_type,n,l)
  z = zero(data_type)
  for j in 1:l
    for i in 1:n
      b[i,j] = i ≤ k ? y.data[j,i] : z
    end
  end
  w = A \ b
  return Interpolator(x,y,view(w,1:k,:),view(w,1+k:n,:),basis,mon)
end

(rbfi::Interpolator)(x::AbstractRealisation) = interpolate(rbfi,x)

function FESpaces.interpolate(rbfi::Interpolator,x::AbstractRealisation)
  k′ = param_length(x)
  l = size(rbfi.rbf_weights,2)
  cache = ConsecutiveParamArray(zeros(l,k′))
  interpolate!(cache,rbfi,x)
  return cache
end

function FESpaces.interpolate!(
  cache::ConsecutiveParamVector,
  rbfi::Interpolator,
  x::Realisation
  )
  k′ = param_length(x)
  l = size(rbfi.rbf_weights,2)

  for j in 1:l
    for i in 1:k′
      rbfji = 0.0
      for q in axes(rbfi.rbf_weights,1)
        rbfji += rbfi.rbf_weights[q,j]*rbfi.rbf_basis(x.params[i],rbfi.x.params[q])
      end

      if !isempty(rbfi.monomial_weights)
        val_poly = rbfi.monomial_basis(x.params[i])
        for (q,val) in enumerate(val_poly)
          rbfji += rbfi.monomial_weights[q,j]*val
        end
      end

      cache.data[j,i] = rbfji
    end
  end
end

# multi field

struct BlockInterpolation{N} <: Interpolation
  interp::Array{<:Interpolation,N}
  touched::Array{Bool,N}

  function BlockInterpolation(
    interp::Array{<:Interpolation,N},
    touched::Array{Bool,N}
    ) where N

    @check size(interp) == size(touched)
    new{N}(interp,touched)
  end
end

Base.ndims(a::BlockInterpolation) = ndims(a.touched)
Base.size(a::BlockInterpolation,args...) = size(a.touched,args...)
Base.axes(a::BlockInterpolation,args...) = axes(a.touched,args...)
Base.length(a::BlockInterpolation) = length(a.touched)
Base.eachindex(a::BlockInterpolation) = eachindex(a.touched)

function Base.getindex(a::BlockInterpolation,i...)
  if !a.touched[i...]
    return nothing
  end
  a.interp[i...]
end

function Base.setindex!(a::BlockInterpolation,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.interp[i...] = v
end

Base.getindex(a::BlockInterpolation,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockInterpolation,v,i::Block) = setindex!(a,v,i.n...)

function Arrays.testitem(a::BlockInterpolation)
  i = findfirst(a.touched)
  @notimplementedif isnothing(i) 
  a.interp[i]
end

function get_cell_idofs(a::BlockInterpolation{N}) where N
  array = Array{Table,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_cell_idofs(a.interp[i])
    end
  end
  return ArrayBlock(array,a.touched)
end

function get_integration_cells(a::BlockInterpolation,args...)
  _union(args...) = @notimplemented
  _union(a::T,b::T) where T<:AbstractVector = union(a,b)
  _union(a::T,b::T) where T<:AppendedArray = lazy_append(union(a.a,b.a),union(a.b,b.b))

  i = findfirst(a.touched)
  isnothing(i) && return Int32[]
  cells = get_integration_cells(a.interp[i],args...)
  for i in 2:length(a)
    if a.touched[i]
      cells = _union(cells,get_integration_cells(a.interp[i],args...))
    end
  end
  return cells
end

function get_owned_icells(a::BlockInterpolation,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function get_owned_icells(a::BlockInterpolation{N},cells::AbstractVector) where N
  array = Array{Vector{Int},N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      array[i] = get_owned_icells(a.interp[i],cells)
    end
  end
  return ArrayBlock(array,a.touched)
end

function move_interpolation(a::BlockInterpolation{N},trial::FESpace,test::FESpace,args...) where N
  I = eltype(a.interp)
  cache = Array{I,N}(undef,size(a))
  for (i,j) in Iterators.product(axes(a)...)
    if a.touched[i,j]
      cache[i,j] = move_interpolation(a.interp[i,j],trial[j],test[i],args...)
    end
  end
  return BlockInterpolation(cache,a.touched)
end

function move_interpolation(a::BlockInterpolation{N},test::FESpace,args...) where N
  I = eltype(a.interp)
  cache = Array{I,N}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = move_interpolation(a.interp[i],test[i],args...)
    end
  end
  return BlockInterpolation(cache,a.touched)
end

# utils

function _get_at_domain(s::Snapshots,rows::AbstractVector{<:Integer})
  data = reshape(get_all_data(s),:,num_params(s))
  _get_at_domain(data,rows)
end

function _get_at_domain(s::SparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = reshape(_all_data(s),:,num_params(s))
  _get_at_domain(data,inds)
end

function _get_at_domain(data::AbstractArray,rows::AbstractVector{<:Integer})
  datav = view(data,rows,:)
  ConsecutiveParamArray(datav)
end

_all_data(s::Snapshots) = get_all_data(s)
_all_data(s::ReshapedSnapshots) = get_all_data(get_param_data(s))
