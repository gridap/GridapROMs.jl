"""
    galerkin_projection(Φₗ,A) -> Any
    galerkin_projection(Φₗ,A,Φᵣ) -> Any

Galerkin projection of `A` on the subspaces specified by a (left, test) subspace `Φₗ`
(row projection) and a (right, trial) subspace `Φᵣ` (column projection)
"""
function galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix)

  proj_basis = basis_left'*basis
  return proj_basis
end

function galerkin_projection(
  basis_left::AbstractMatrix{S},
  basis::ParamSparseMatrix{T},
  basis_right::AbstractMatrix{S}
  ) where {T,S}

  @check size(basis,1) == size(basis,2)
  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)

  TS = promote_type(T,S)
  proj_basis = zeros(TS,nleft,n,nright)
  @inbounds for i = 1:n
    @views proj_basis[:,i,:] = basis_left'*param_getindex(basis,i)*basis_right
  end

  return proj_basis
end
