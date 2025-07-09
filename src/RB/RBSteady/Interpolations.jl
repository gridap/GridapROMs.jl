abstract type Interpolation end

get_interpolation(a::Interpolation) = @abstractmethod

Interpolation(::MDEIMReduction,args...) = MDEIMInterpolation(args...)
Interpolation(::RBFReduction,args...) = RBFInterpolation(args...)

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,x::Any)
  @abstractmethod
end

function reduced_triangulation(trian::Triangulation,a::Interpolation)
  return trian
end

# EIM interpolation

struct MDEIMInterpolation{A,B} <: Interpolation
  interpolation::A
  domain::B
end

MDEIMInterpolation() = MDEIMInterpolation(nothing,nothing)

function MDEIMInterpolation(basis::Projection,trian::Triangulation,test::RBSpace)
  rows,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = vector_domain(trian,test,rows)
  MDEIMInterpolation(interp,domain)
end

function MDEIMInterpolation(basis::Projection,trian::Triangulation,trial::RBSpace,test::RBSpace)
  rows,interp = empirical_interpolation(basis)
  factor = lu(interp)
  domain = matrix_domain(trian,trial,test,rows,cols)
  MDEIMInterpolation(interp,domain)
end

get_interpolation(a::MDEIMInterpolation) = a.interpolation

get_integration_domain(a::MDEIMInterpolation) = a.domain
get_integration_cells(a::MDEIMInterpolation,args...) = get_integration_cells(get_integration_domain(a),args...)
get_cellids_rows(a::MDEIMInterpolation) = get_cellids_rows(get_integration_domain(a))
get_cellids_cols(a::MDEIMInterpolation) = get_cellids_cols(get_integration_domain(a))
get_owned_icells(a::MDEIMInterpolation,args...) = get_owned_icells(a,get_integration_cells(a,args...))
get_owned_icells(a::MDEIMInterpolation,cells::AbstractVector) = get_owned_icells(get_integration_domain(a),cells)

function FESpaces.interpolate!(cache::AbstractArray,a::MDEIMInterpolation,b::AbstractArray)
  ldiv!(cache,a,b)
  cache
end

function reduced_triangulation(trian::Triangulation,a::MDEIMInterpolation)
  red_cells = get_integration_cells(a)
  red_trian = view(trian,red_cells)
  return red_trian
end

# RBF interpolation

struct RBFInterpolation{A} <: Interpolation
  interpolation::A
end

RBFInterpolation() = RBFInterpolation(nothing)

function RBFInterpolation(red::RBFHyperReduction,a::Projection,s::Snapshots)
  RBFInterpolation(interp_strategy(red),a,s)
end

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

function FESpaces.interpolate!(cache::AbstractArray,a::RBFInterpolation,r::AbstractRealization)
  interpolate!(cache,get_interpolation(a),r)
  cache
end

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
  factor = factorize(A)
  l = innerlength(y)
  w = zeros(data_type,n,l)
  b = zeros(data_type,n,l)
  z = zero(data_type)
  for j in 1:l
    for i in 1:n
      b[i,j] = i < k ? y.data[j,i] : z
    end
  end
  ldiv!(w,factor,b)
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

function FESpaces.interpolate!(cache::ConsecutiveParamVector,rbfi::Interpolator,x::AbstractRealization)
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
  datav = view(data,rows,:)
  ConsecutiveParamArray(datav)
end

function get_at_domain(s::SparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  data = get_all_data(s)
  datav = view(data,inds,:)
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
  a.array[i...]
end

function Base.setindex!(a::BlockInterpolation,v,i...)
  @check a.touched[i...] "Only touched entries can be set"
  a.array[i...] = v
end

Base.getindex(a::BlockInterpolation,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockInterpolation,v,i::Block) = setindex!(a,v,i.n...)

function Arrays.testitem(a::BlockInterpolation)
  i = findall(a.touched)
  @notimplementedif length(i) == 0
  a.array[first(i)]
end

for f in (:get_cellids_rows,:get_cellids_cols)
  @eval begin
    function Arrays.return_cache(::typeof($f),a::BlockInterpolation)
      block_cache = Array{Table,ndims(a)}(undef,size(a))
      return block_cache
    end

    function $f(a::BlockInterpolation)
      cache = return_cache($f,a)
      for i in eachindex(a)
        if a.touched[i]
          cache[i] = $f(a[i])
        end
      end
      return ArrayBlock(cache,a.touched)
    end
  end
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

const AbstractMDEIMInterp = Union{MDEIMInterpolation,BlockInterpolation{<:MDEIMInterpolation}}

const AbstractRBFInterp = Union{RBFInterpolation,BlockInterpolation{<:RBFInterpolation}}
