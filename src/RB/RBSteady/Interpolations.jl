abstract type Interpolation end

get_interpolation(a::Interpolation) = @abstractmethod
get_integration_domain(a::Interpolation) = @abstractmethod
get_integration_cells(a::Interpolation,args...) = get_integration_cells(get_integration_domain(a),args...)
get_cell_idofs(a::Interpolation) = get_cell_idofs(get_integration_domain(a))
get_owned_icells(a::Interpolation,args...) = get_owned_icells(a,get_integration_cells(a,args...))
get_owned_icells(a::Interpolation,cells::AbstractVector) = get_owned_icells(get_integration_domain(a),cells)

Interpolation(red::MDEIMHyperReduction,args...) = MDEIMInterpolation(args...)
Interpolation(red::SOPTHyperReduction,args...) = SOPTInterpolation(args...)
Interpolation(red::RBFHyperReduction,args...) = RBFInterpolation(interp_strategy(red),args...)

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,x::Any)
  @abstractmethod
end

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,b::AbstractArray)
  ldiv!(cache,get_interpolation(a),b)
  cache
end

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,r::AbstractRealization)
  interpolate!(cache,get_interpolation(a),r)
  cache
end

function reduced_triangulation(trian::Triangulation,a::Interpolation)
  return trian
end

# Empty interpolation

struct EmptyInterpolation <: Interpolation end

# EIM interpolation

struct MDEIMInterpolation{A,B<:IntegrationDomain} <: Interpolation
  interpolation::A
  domain::B
end

MDEIMInterpolation() = EmptyInterpolation()
SOPTInterpolation() = EmptyInterpolation()

for (f,g) in zip((:MDEIMInterpolation,:SOPTInterpolation),(:empirical_interpolation,:s_opt))
  @eval begin
    function $f(basis::Projection,trian::Triangulation,test::RBSpace)
      rows,interp = $g(basis)
      factor = lu(interp)
      domain = IntegrationDomain(trian,test,rows)
      MDEIMInterpolation(factor,domain)
    end

    function $f(basis::Projection,trian::Triangulation,trial::RBSpace,test::RBSpace)
      (rows,cols),interp = $g(basis)
      factor = lu(interp)
      domain = IntegrationDomain(trian,trial,test,rows,cols)
      MDEIMInterpolation(factor,domain)
    end
  end
end

get_interpolation(a::MDEIMInterpolation) = a.interpolation
get_integration_domain(a::MDEIMInterpolation) = a.domain

function reduced_triangulation(trian::Triangulation,a::MDEIMInterpolation)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end

function move_interpolation(a::MDEIMInterpolation,args...)
  interpolation = get_interpolation(a)
  domain = move_integration_domain(get_integration_domain(a),args...)
  MDEIMInterpolation(interpolation,domain)
end

# RBF interpolation

struct RBFInterpolation{A<:Interpolator} <: Interpolation
  interpolation::A
end

RBFInterpolation(strategy::AbstractRadialBasis) = EmptyInterpolation()

function RBFInterpolation(strategy::AbstractRadialBasis,a::Projection,s::Snapshots)
  inds,interp = empirical_interpolation(a)
  factor = lu(interp)
  r = get_realization(s)
  red_data = get_at_domain(s,inds)
  coeff = allocate_coefficient(a,r)
  ldiv!(coeff,factor,red_data)
  interp = Interpolator(r,coeff,strategy)
  RBFInterpolation(interp)
end

get_interpolation(a::RBFInterpolation) = a.interpolation

function RadialBasisFunctions.Interpolator(
  x::Realization,
  y::ConsecutiveParamArray,
  basis::B=PHS()
  ) where B<:AbstractRadialBasis

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
      b[i,j] = i < k ? y.data[j,i] : z
    end
  end
  w = A \ b
  return Interpolator(x,y,view(w,1:k,:),view(w,1+k:n,:),basis,mon)
end

(rbfi::Interpolator)(x::AbstractRealization) = interpolate(rbfi,x)

function FESpaces.interpolate(rbfi::Interpolator,x::AbstractRealization)
  k′ = param_length(x)
  l = size(rbfi.rbf_weights,2)
  cache = ConsecutiveParamArray(zeros(l,k′))
  interpolate!(cache,rbfi,x)
  return cache
end

function FESpaces.interpolate!(cache::ConsecutiveParamVector,rbfi::Interpolator,x::Realization)
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

# interpolation utils

function get_at_domain(s::Snapshots,rows::AbstractVector{<:Integer})
  data = reshape(get_all_data(s),:,num_params(s))
  get_at_domain(data,rows)
end

function get_at_domain(s::SparseSnapshots,rowscols::Tuple)
  _all_data(s::Snapshots) = get_all_data(s)
  _all_data(s::ReshapedSnapshots) = get_all_data(get_param_data(s))
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = reshape(_all_data(s),:,num_params(s))
  get_at_domain(data,inds)
end

function get_at_domain(data::AbstractArray,rows::AbstractVector{<:Integer})
  datav = view(data,rows,:)
  ConsecutiveParamArray(datav)
end

# multi field

struct BlockInterpolation{I<:Interpolation,N} <: Interpolation
  interp::Array{I,N}
  touched::Array{Bool,N}
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
  i = findall(a.touched)
  @notimplementedif length(i) == 0
  a.interp[first(i)]
end

function Arrays.return_cache(::typeof(get_cell_idofs),a::BlockInterpolation)
  block_cache = Array{Table,ndims(a)}(undef,size(a))
  return block_cache
end

function get_cell_idofs(a::BlockInterpolation)
  cache = return_cache(get_cell_idofs,a)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_cell_idofs(a[i])
    end
  end
  return ArrayBlock(cache,a.touched)
end

function Arrays.return_cache(::typeof(get_integration_cells),a::BlockInterpolation,args...)
  ntouched = length(findall(a.touched))
  cache = get_integration_cells(testitem(a),args...)
  block_cache = Vector{typeof(cache)}(undef,ntouched)
  return block_cache
end

function get_integration_cells(a::BlockInterpolation,args...)
  _union(a) = a
  _union(a,b) = union(a,b)
  _union(a::AppendedArray,b::AppendedArray) = lazy_append(union(a.a,b.a),union(a.b,b.b))
  _union(a,b,c...) = _union(_union(a,b),c...)

  cache = return_cache(get_integration_cells,a,args...)
  count = 0
  for i in eachindex(a)
    if a.touched[i]
      count += 1
      cache[count] = get_integration_cells(a[i],args...)
    end
  end
  return _union(cache...)
end

function get_owned_icells(a::BlockInterpolation,args...)
  cells = get_integration_cells(a,args...)
  get_owned_icells(a,cells)
end

function Arrays.return_cache(::typeof(get_owned_icells),a::BlockInterpolation,cells)
  cache = get_owned_icells(testitem(a),cells)
  block_cache = Array{typeof(cache),ndims(a)}(undef,size(a))
  return block_cache
end

function get_owned_icells(a::BlockInterpolation,cells::AbstractVector)
  cache = return_cache(get_owned_icells,a,cells)
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = get_owned_icells(a[i],cells)
    end
  end
  return ArrayBlock(cache,a.touched)
end

function move_interpolation(a::BlockInterpolation,trial::FESpace,test::FESpace,args...)
  I = typeof(testitem(a))
  cache = Array{I,ndims(a)}(undef,size(a))
  for (i,j) in Iterators.product(axes(a)...)
    if a.touched[i,j]
      cache[i,j] = move_interpolation(a[i,j],trial[j],test[i],args...)
    end
  end
  return BlockInterpolation(cache,a.touched)
end

function move_interpolation(a::BlockInterpolation,test::FESpace,args...)
  I = typeof(testitem(a))
  cache = Array{I,ndims(a)}(undef,size(a))
  for i in eachindex(a)
    if a.touched[i]
      cache[i] = move_interpolation(a[i],test[i],args...)
    end
  end
  return BlockInterpolation(cache,a.touched)
end

function reduced_triangulation(trian::Triangulation,a::BlockInterpolation{<:MDEIMInterpolation})
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end
