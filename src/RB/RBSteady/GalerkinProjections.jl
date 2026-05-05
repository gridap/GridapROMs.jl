"""
    galerkin_projection(Φₗ,A) -> Any
    galerkin_projection(Φₗ,A,Φᵣ) -> Any

Galerkin projection of `A` on the subspaces specified by a (left, test) subspace `Φₗ`
(row projection) and a (right, trial) subspace `Φᵣ` (column projection)
"""
function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix
  )

  proj_basis = basis_left'*basis
  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractParamVector
  )

  galerkin_projection(basis_left,get_all_data(basis))
end

function galerkin_projection(
  basis_left::AbstractMatrix{S},
  basis::ParamSparseMatrix{T},
  basis_right::AbstractMatrix{S}
  ) where {T,S}

  TS = promote_type(T,S)
  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)
  proj_basis = zeros(TS,nleft,n,nright)

  @inbounds for i = 1:n
    @views proj_basis[:,i,:] = basis_left'*param_getindex(basis,i)*basis_right
  end

  return proj_basis
end

# not really in-place 

function galerkin_projection!(
  cache::AbstractMatrix{<:Number},
  basis_left,
  basis,
  args...
  )

  proj_basis = galerkin_projection(basis_left,basis,args...)
  @check size(cache) == size(proj_basis)
  copyto!(cache,proj_basis)
  return cache
end

function galerkin_projection!(
  cache::AbstractArray{<:Number,3},
  basis_left,
  basis,
  basis_right,
  args...
  )

  proj_basis = galerkin_projection(basis_left,basis,basis_right,args...)
  @check ndims(proj_basis) == 3
  @check size(cache,1) == size(proj_basis,1)
  @check size(cache,2) == size(proj_basis,3)
  @check size(cache,3) == size(proj_basis,2)
  @inbounds for i in eachindex(cache)
    cache[:,:,i] = proj_basis[:,i,:]
  end
  return cache
end

function galerkin_projection!(cache::AbstractParamArray,args...)
  galerkin_projection!(get_all_data(cache),args...)
end