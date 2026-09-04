"""
    galerkin_projection(Φₗ,A) -> Any
    galerkin_projection(Φₗ,A,Φᵣ) -> Any

Galerkin projection of `A` on the subspaces specified by a (left, test) subspace `Φₗ`
(row projection) and a (right, trial) subspace `Φᵣ` (column projection)
"""
function galerkin_projection(
  basis_left::AbstractMatrix,
  a::AbstractMatrix
  )

  proj_basis = basis_left'*a
  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  a::AbstractParamVector
  )

  galerkin_projection(basis_left,get_all_data(a))
end

function galerkin_projection(
  basis_left::AbstractMatrix{S},
  a::AbstractParamMatrix{T},
  basis_right::AbstractMatrix{S}
  ) where {T,S}

  TS = promote_type(T,S)
  nleft = size(basis_left,2)
  n = size(a,1)
  nright = size(basis_right,2)
  proj_basis = zeros(TS,nleft,n,nright)

  @inbounds for i = 1:n
    @views proj_basis[:,i,:] = basis_left'*param_getindex(a,i)*basis_right
  end

  return proj_basis
end

# not really in-place 

function galerkin_projection!(
  cache,
  basis_left,
  a,
  args...
  )

  proj_basis = galerkin_projection(basis_left,a,args...)
  copy_projection!(cache,proj_basis)
  return cache
end

function copy_projection!(cache,proj_basis)
  @abstractmethod
end

function copy_projection!(
  cache::AbstractParamVector,
  proj_basis::AbstractMatrix{<:Number}
  )

  data = get_all_data(cache)
  @check size(data) == size(proj_basis)
  copyto!(data,proj_basis)
end

function copy_projection!(
  cache::AbstractParamMatrix,
  proj_basis::AbstractArray{<:Number,3}
  )

  data = get_all_data(cache)
  @check size(data,1) == size(proj_basis,1)
  @check size(data,2) == size(proj_basis,3)
  @check size(data,3) == size(proj_basis,2)
  @inbounds @views for i in axes(proj_basis,2)
    data[:,:,i] = proj_basis[:,i,:]
  end
end

function copy_projection!(
  cache::ArrayBlock{A,N},
  proj_basis::ArrayBlock{B,N}
  ) where {A,B,N}

  @check size(cache) == size(proj_basis)
  for i in eachindex(cache)
    if cache.touched[i] && proj_basis.touched[i]
      copy_projection!(cache[i],proj_basis[i])
    end
  end
end