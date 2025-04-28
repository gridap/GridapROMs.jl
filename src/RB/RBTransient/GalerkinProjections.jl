function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::Function)

  @notimplemented
end

function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  basis_right::AbstractMatrix,
  combine::Function)

  nleft = size(basis_left,2)
  n = size(basis,2)
  nright = size(basis_right,2)

  proj_basis = zeros(nleft,n,nright)
  proj_basis′ = copy(proj_basis)

  @inbounds for i = 1:nleft, k = 1:n, j = 1:nright
    s,s′ = 0,0
    for α = axes(basis,1)
      s += basis_left[α,i]*basis[α,k]*basis_right[α,j]
      if α < size(basis,1)
        s′ += basis_left[α+1,i]*basis[α+1,k]*basis_right[α,j]
      end
    end
    proj_basis[i,k,j] = s
    proj_basis′[i,k,j] = s′
  end

  combine(proj_basis,proj_basis′)
end

function RBSteady.galerkin_projection(
  core_left::AbstractArray{T,3},
  basis::AbstractMatrix,
  core_right::AbstractArray{S,3},
  combine::Function
  ) where {T,S}

  s1,s2,s3 = size(core_left)
  s4,s5,s6 = size(core_right)
  @check s2 == s5
  core = reshape(basis,:,s2,size(basis,2))
  contraction(core_left,core,core_right,combine)
end
