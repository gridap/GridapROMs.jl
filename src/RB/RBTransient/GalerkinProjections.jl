function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::TimeCombination
  )

  galerkin_projection(basis_left,basis)
end

function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix{S},
  basis::AbstractMatrix{T},
  basis_right::AbstractMatrix{S},
  combine::TimeCombination
  ) where {T,S}

  nleft = size(basis_left,2)
  Nt,n = size(basis)
  nright = size(basis_right,2)

  θ = get_coefficients(combine,Nt)

  TS = promote_type(T,S)
  proj_basis = zeros(TS,nleft,n,nright)

  @inbounds for i = 1:nleft, k = 1:n, j = 1:nright
    s = zero(TS)
    for γ = eachindex(θ)
      for α = axes(basis,1)
        α+γ > Nt+1 && break 
        s += θ[γ]*basis_left[α+γ-1,i]*basis[α+γ-1,k]*basis_right[α,j]
      end
    end
    proj_basis[i,k,j] = s
  end

  return proj_basis
end

function RBSteady.galerkin_projection(
  core_left::AbstractArray{T,3},
  basis::AbstractMatrix,
  core_right::AbstractArray{T,3},
  combine::TimeCombination
  ) where T

  _,s2,_ = size(core_left)
  _,s5,_ = size(core_right)
  @check s2 == s5
  core = reshape(basis,:,s2,size(basis,2))
  contraction(core_left,core,core_right,combine)
end
