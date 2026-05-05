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

function galerkin_projection!(
  proj_basis::AbstractMatrix,
  basis_left::AbstractMatrix,
  basis::AbstractMatrix
  )

  @check size(proj_basis) == (size(basis_left,2),size(basis,2))
  mul!(proj_basis,basis_left',basis)
  return proj_basis
end

function galerkin_projection!(
  proj_basis::AbstractParamVector,
  basis_left::AbstractMatrix,
  basis::AbstractParamVector
  )

  galerkin_projection!(get_all_data(proj_basis),basis_left,get_all_data(basis))
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
  galerkin_projection!(proj_basis,basis_left,basis,basis_right)
end

function galerkin_projection!(
  proj_basis::AbstractArray,
  basis_left::AbstractMatrix,
  basis::ParamSparseMatrix,
  basis_right::AbstractMatrix
  ) 

  @check size(basis,1) == size(basis,2)
  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)
  @check size(proj_basis) == (nleft,n,nright)

  @inbounds for i = 1:n
    @views proj_basis[:,i,:] = basis_left'*param_getindex(basis,i)*basis_right
  end

  return proj_basis
end

function galerkin_projection!(
  proj_basis::AbstractParamMatrix,
  basis_left::AbstractMatrix,
  basis::ParamSparseMatrix,
  basis_right::AbstractMatrix
  ) 

  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)
  cache = get_all_data(proj_basis)
  if size(cache) == (nleft,n,nright)
    galerkin_projection!(cache,basis_left,basis,basis_right)
  else
    @check size(cache) == (nleft,nright,n)
    @inbounds for i = 1:n
      @views cache[:,:,i] = basis_left'*param_getindex(basis,i)*basis_right
    end
  end
  return proj_basis
end