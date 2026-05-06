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
  @inbounds @views for i in axes(proj_basis,2)
    cache[:,:,i] = proj_basis[:,i,:]
  end
  return cache
end

function galerkin_projection!(cache::AbstractParamArray,args...)
  galerkin_projection!(get_all_data(cache),args...)
end

# multi-field interface 

function galerkin_projection(
  basis_left::VectorBlock,
  basis::BlockParamVector
  ) 

  block_cache = Vector{Any}(undef,length(basis_left))
  for i in eachindex(basis_left)
    if basis_left.touched[i]
      block_cache[i] = galerkin_projection(basis_left[i],blocks(basis)[i])
    end
  end
  return ArrayBlock(block_cache,basis_left.touched)
end

function galerkin_projection(
  basis_left::VectorBlock,
  basis::BlockParamVector,
  basis_right::VectorBlock,
  args...
  ) 

  block_cache = Matrix{Any}(undef,length(basis_left),length(basis_right))
  touched = Matrix{Bool}(undef,length(basis_left),length(basis_right))
  for i in eachindex(basis_left), j in eachindex(basis_right)
    touched[i,j] = basis_left.touched[i] && basis_right.touched[j]
    if touched[i,j]
      block_cache[i,j] = galerkin_projection(basis_left[i],blocks(basis)[i,j],basis_right[j])
    end
  end
  return ArrayBlock(block_cache,touched)
end

function galerkin_projection!(
  cache::VectorBlock,
  basis_left::VectorBlock,
  basis::BlockParamVector
  )

  for i in 1:size(cache,1)
    if cache.touched[i]
      galerkin_projection!(cache[i],basis_left[i],blocks(basis)[i])
    end
  end
  return cache
end

function galerkin_projection!(
  cache::MatrixBlock,
  basis_left::VectorBlock,
  basis::BlockParamMatrix,
  basis_right::VectorBlock,
  args...
  )

  for i in 1:size(cache,1), j in 1:size(cache,2)
    if cache.touched[i,j]
      galerkin_projection!(cache[i,j],basis_left[i],blocks(basis)[i,j],basis_right[j])
    end
  end
  return cache
end

function galerkin_projection(
  basis_left::VectorBlock,
  basis::VectorBlock
  ) 

  block_cache = Vector{Any}(undef,length(basis_left))
  for i in eachindex(basis_left)
    if basis_left.touched[i]
      @check basis.touched[i]
      block_cache[i] = galerkin_projection(basis_left[i],basis[i])
    end
  end
  return ArrayBlock(block_cache,basis_left.touched)
end

function galerkin_projection(
  basis_left::VectorBlock,
  basis::MatrixBlock,
  basis_right::VectorBlock,
  args...
  ) 

  block_cache = Matrix{Any}(undef,length(basis_left),length(basis_right))
  touched = Matrix{Bool}(undef,length(basis_left),length(basis_right))
  for i in eachindex(basis_left), j in eachindex(basis_right)
    touched[i,j] = basis_left.touched[i] && basis_right.touched[j]
    if touched[i,j]
      @check basis.touched[i,j]
      block_cache[i,j] = galerkin_projection(basis_left[i],basis[i,j],basis_right[j])
    end
  end
  return ArrayBlock(block_cache,touched)
end

function galerkin_projection!(
  cache::VectorBlock,
  basis_left::VectorBlock,
  basis::VectorBlock
  )

  for i in 1:size(cache,1)
    if cache.touched[i]
      @check basis.touched[i]
      galerkin_projection!(cache[i],basis_left[i],blocks(basis)[i])
    end
  end
  return cache
end

function galerkin_projection!(
  cache::MatrixBlock,
  basis_left::VectorBlock,
  basis::MatrixBlock,
  basis_right::VectorBlock,
  args...
  )

  for i in 1:size(cache,1), j in 1:size(cache,2)
    if cache.touched[i,j]
      @check basis.touched[i,j]
      galerkin_projection!(cache[i,j],basis_left[i],blocks(basis)[i,j],basis_right[j])
    end
  end
  return cache
end