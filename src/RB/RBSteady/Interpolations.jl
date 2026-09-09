abstract type Interpolation end

Interpolation(args...) = @abstractmethod

get_integration_cells(a::Interpolation) = Int32[]
get_cell_idofs(a::Interpolation) = empty_table(Int,Int32,0)
get_owned_icells(a::Interpolation) = collect(1:length(get_integration_cells(a)))
get_interpolation_dofs(a::Interpolation) = @abstractmethod

function FESpaces.interpolate!(cache::AbstractArray,a::Interpolation,x::Any)
  cache
end

function reduced_triangulation(trian::Triangulation,a::Interpolation)
  red_cells = get_integration_cells(a)
  red_trian = ChildTriangulation(trian,red_cells)
  return red_trian
end

function get_owned_icells(a::Interpolation,trian::Triangulation) 
  cells = get_integration_cells(trian)
  get_owned_icells(a,cells)
end

function get_owned_icells(a::Interpolation,cells::AbstractVector)
  cellsi = get_integration_cells(a)
  icells = filter(!isnothing,indexin(cellsi,cells))
  Int.(icells)
end

# Empty interpolation

struct EmptyInterpolation <: Interpolation end

function Interpolation(red::HyperReduction)
  EmptyInterpolation()
end

struct FullInterpolation <: Interpolation
  cells::Vector{Int32}
end

function Interpolation(red::NoHyperReduction,trian,args...)
  n = num_cells(trian)
  cells = collect(Int32,1:n)
  FullInterpolation(cells)
end

get_integration_cells(a::FullInterpolation) = a.cells

# EIM interpolation

struct GreedyInterpolation{A,B<:IntegrationDomain} <: Interpolation
  interpolation::A
  domain::B
end

for (T,f) in zip((:DEIMHyperReduction,:SOPTHyperReduction),(:DEIM,:SOPT))
  @eval begin
    function Interpolation(red::$T,a::Projection,trian,test)
      rows,interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(trian,test,rows)
      GreedyInterpolation(factor,domain)
    end

    function Interpolation(red::$T,a::Projection,trian,trial,test)
      (rows,cols),interp = $f(a)
      factor = lu(interp)
      domain = IntegrationDomain(trian,trial,test,rows,cols)
      GreedyInterpolation(factor,domain)
    end
  end
end

get_integration_cells(a::GreedyInterpolation) = get_integration_cells(a.domain)
get_cell_idofs(a::GreedyInterpolation) = get_cell_idofs(a.domain)
get_interpolation_dofs(a::GreedyInterpolation) = get_interpolation_dofs(a.domain)

function FESpaces.interpolate!(cache::AbstractArray,a::GreedyInterpolation,b::AbstractArray)
  ldiv!(cache,a.interpolation,b)
  cache
end

# RBF interpolation

struct RBFInterpolation{A<:Interpolator} <: Interpolation
  interpolation::A
end

function Interpolation(red::RBFHyperReduction,a::Projection,s::Snapshots)
  strategy = interp_strategy(red)
  inds,interp = DEIM(a)
  factor = lu(interp)
  r = get_realisation(s)
  red_data = get_at_domain(s,inds)
  coeff = parameterise(allocate_in_domain(a),r)
  ldiv!(coeff,factor,red_data)
  interp = Interpolator(r,coeff,strategy)
  RBFInterpolation(interp)
end

function FESpaces.interpolate!(cache::AbstractArray,a::RBFInterpolation,r::AbstractRealisation)
  interpolate!(cache,a.interpolation,get_params(r))
  cache
end

function Interpolator(
  x::Realisation,
  y::ConsecutiveParamArray,
  basis::AbstractRadialBasis=PHS()
  ) 

  dim = length(first(x))
  k = param_length(x)
  
  # Reduce polynomial degree to ensure non-singular collocation system
  poly_deg = basis.poly_deg
  npoly = binomial(dim + poly_deg,poly_deg)
  while k < npoly && poly_deg > 0
    poly_deg -= 1
    npoly = binomial(dim + poly_deg,poly_deg)
  end
  
  if poly_deg < basis.poly_deg
    @warn "RBF cluster too small for polynomial degree; reducing from $(basis.poly_deg) to $poly_deg" k npoly dim
  end
  
  n = k + npoly
  mon = MonomialBasis(dim,poly_deg)
  data_type = promote_type(eltype(first(x.params)),eltype2(y))
  A = Symmetric(zeros(data_type,n,n))
  _build_collocation_matrix!(A,x.params,basis,mon,k)
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
  T = eltype(rbfi.rbf_weights)
  cache = ConsecutiveParamArray(zeros(T,l,k′))
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

function get_at_domain(s::Snapshots,rows::AbstractVector)
  data = flatten(s)
  get_at_domain(data,rows)
end

function get_at_domain(s::SparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  sparsity = get_sparsity(get_dof_map(s))
  inds = sparsify_split_indices(rows,cols,sparsity)
  get_at_domain(s,inds)
end

function get_at_domain(data::AbstractArray,rows::AbstractVector)
  datav = view(data,rows,:)
  ConsecutiveParamArray(datav)
end

# multi field

struct BlockInterpolation{N} <: Interpolation
  interp::Array{<:Interpolation,N}
end

Base.ndims(a::BlockInterpolation) = ndims(a.interp)
Base.size(a::BlockInterpolation,args...) = size(a.interp,args...)
Base.axes(a::BlockInterpolation,args...) = axes(a.interp,args...)
Base.length(a::BlockInterpolation) = length(a.interp)
Base.eachindex(a::BlockInterpolation) = eachindex(a.interp)

Base.getindex(a::BlockInterpolation,i...) = a.interp[i...]
Base.setindex!(a::BlockInterpolation,v,i...) = (a.interp[i...] = v)

Arrays.testitem(a::BlockInterpolation) = first(a.interp)

Base.getindex(a::BlockInterpolation,i::Block) = getindex(a,i.n...)
Base.setindex!(a::BlockInterpolation,v,i::Block) = setindex!(a,v,i.n...)

function get_cell_idofs(a::BlockInterpolation{N}) where N
  map(get_cell_idofs,a.interp)
end

function get_integration_cells(a::BlockInterpolation)
  _union(args...) = @notimplemented
  _union(a::T,b::T) where T<:AbstractVector = union(a,b)
  _union(a::T,b::T) where T<:AppendedArray = lazy_append(union(a.a,b.a),union(a.b,b.b))

  cells = get_integration_cells(a.interp[1])
  for i in 2:length(a)
    cells = _union(cells,get_integration_cells(a.interp[i]))
  end
  return cells
end

function get_owned_icells(a::BlockInterpolation{N},cells::AbstractVector) where N
  map(itp -> get_owned_icells(itp,cells),a.interp)
end