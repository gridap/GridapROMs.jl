function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::TimeCombination
  )

  galerkin_projection(basis_left,basis)
end

function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix,
  basis::AbstractParamVector,
  combine::TimeCombination
  )

  galerkin_projection(basis_left,get_all_data(basis),combine)
end

function RBSteady.galerkin_projection(
  basis_left::AbstractMatrix{S},
  basis::AbstractMatrix{T},
  basis_right::AbstractMatrix{S},
  combine::TimeCombination
  ) where {T,S}

  nleft = size(basis_left,2)
  n = size(basis,2)
  nright = size(basis_right,2)
  TS = promote_type(T,S)
  proj_basis = zeros(TS,nleft,n,nright)
  galerkin_projection!(proj_basis,basis_left,basis,basis_right,combine)
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

function RBSteady.galerkin_projection!(
  proj_basis::AbstractMatrix,
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  combine::TimeCombination
  )

  galerkin_projection!(proj_basis,basis_left,basis)
  return proj_basis
end

function RBSteady.galerkin_projection!(
  proj_basis::AbstractParamVector,
  basis_left::AbstractMatrix,
  basis::AbstractParamVector,
  combine::TimeCombination
  )

  galerkin_projection!(get_all_data(proj_basis),basis_left,get_all_data(basis),combine)
  return proj_basis
end

function RBSteady.galerkin_projection!(
  proj_basis::AbstractArray{T,3} where T,
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  basis_right::AbstractMatrix,
  combine::TimeCombination
  )

  nleft = size(basis_left,2)
  Nt,n = size(basis)
  @check size(basis_left,1) == Nt
  @check size(basis_right,1) == Nt
  nright = size(basis_right,2)
  @check size(proj_basis) == (nleft,n,nright)

  θ = get_coefficients(combine,Nt)

  @inbounds for i = 1:nleft, k = 1:n, j = 1:nright
    s = zero(T)
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

function galerkin_projection!(
  proj_basis::AbstractParamMatrix,
  basis_left::AbstractMatrix,
  basis::AbstractMatrix,
  basis_right::AbstractMatrix,
  combine::TimeCombination
  ) 

  nleft = size(basis_left,2)
  n = size(basis,1)
  nright = size(basis_right,2)
  cache = get_all_data(proj_basis)
  if size(cache) == (nleft,n,nright)
    galerkin_projection!(cache,basis_left,basis,basis_right,combine)
  else
    @check size(cache) == (nleft,nright,n)
    @check size(basis_left,1) == Nt
    @check size(basis_right,1) == Nt

    θ = get_coefficients(combine,Nt)

    @inbounds for i = 1:nleft, j = 1:nright, k = 1:n
      s = zero(T)
      for γ = eachindex(θ)
        for α = axes(basis,1)
          α+γ > Nt+1 && break 
          s += θ[γ]*basis_left[α+γ-1,i]*basis[α+γ-1,k]*basis_right[α,j]
        end
      end
      cache[i,j,k] = s
    end
  end
  return proj_basis
end

function RBSteady.galerkin_projection!(
  proj_basis::AbstractArray,
  core_left::AbstractArray{T,3},
  basis::AbstractMatrix,
  core_right::AbstractArray{T,3},
  combine::TimeCombination
  ) where T

  _,s2,_ = size(core_left)
  _,s5,_ = size(core_right)
  @check s2 == s5
  core = reshape(basis,:,s2,size(basis,2))
  contraction!(proj_basis,core_left,core,core_right,combine)
  return proj_basis
end