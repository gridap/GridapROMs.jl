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
  a::ParamSparseMatrix{T},
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
  cache::AbstractParamVector,
  basis_left,
  a,
  args...
  )

  data = get_all_data(cache)
  proj_basis = galerkin_projection(basis_left,a,args...)
  @check size(data) == size(proj_basis)
  copyto!(data,proj_basis)
  return cache
end

function galerkin_projection!(
  cache::AbstractParamMatrix,
  basis_left,
  a,
  basis_right,
  args...
  )

  data = get_all_data(cache)
  proj_basis = galerkin_projection(basis_left,a,basis_right,args...)
  @check ndims(proj_basis) == 3
  @check size(data,1) == size(proj_basis,1)
  @check size(data,2) == size(proj_basis,3)
  @check size(data,3) == size(proj_basis,2)
  @inbounds @views for i in axes(proj_basis,2)
    data[:,:,i] = proj_basis[:,i,:]
  end
  return cache
end

# multi-field interface 

# function galerkin_projection(
#   basis_left::VectorBlock,
#   a::BlockParamVector,
#   args...
#   ) 

#   block_cache = Vector{Any}(undef,length(basis_left))
#   for i in eachindex(basis_left)
#     if basis_left.touched[i]
#       block_cache[i] = galerkin_projection(basis_left[i],blocks(a)[i],args...)
#     end
#   end
#   return ArrayBlock(block_cache,basis_left.touched)
# end

# function galerkin_projection(
#   basis_left::VectorBlock,
#   a::BlockParamVector,
#   basis_right::VectorBlock,
#   args...
#   ) 

#   block_cache = Matrix{Any}(undef,length(basis_left),length(basis_right))
#   touched = Matrix{Bool}(undef,length(basis_left),length(basis_right))
#   for i in eachindex(basis_left), j in eachindex(basis_right)
#     touched[i,j] = basis_left.touched[i] && basis_right.touched[j]
#     if touched[i,j]
#       block_cache[i,j] = galerkin_projection(basis_left[i],blocks(a)[i,j],basis_right[j],args...)
#     end
#   end
#   return ArrayBlock(block_cache,touched)
# end

# function galerkin_projection!(
#   cache::VectorBlock,
#   basis_left::VectorBlock,
#   a::BlockParamVector,
#   args...
#   )

#   for i in 1:size(cache,1)
#     if cache.touched[i]
#       galerkin_projection!(cache[i],basis_left[i],blocks(a)[i],args...)
#     end
#   end
#   return cache
# end

# function galerkin_projection!(
#   cache::MatrixBlock,
#   basis_left::VectorBlock,
#   a::BlockParamMatrix,
#   basis_right::VectorBlock,
#   args...
#   )

#   for i in 1:size(cache,1), j in 1:size(cache,2)
#     if cache.touched[i,j]
#       galerkin_projection!(cache[i,j],basis_left[i],blocks(a)[i,j],basis_right[j],args...)
#     end
#   end
#   return cache
# end

# function galerkin_projection(
#   basis_left::VectorBlock,
#   a::VectorBlock,
#   args...
#   ) 

#   block_cache = Vector{Any}(undef,length(basis_left))
#   for i in eachindex(basis_left)
#     if basis_left.touched[i]
#       @check a.touched[i]
#       block_cache[i] = galerkin_projection(basis_left[i],a[i],args...)
#     end
#   end
#   return ArrayBlock(block_cache,basis_left.touched)
# end

# function galerkin_projection(
#   basis_left::VectorBlock,
#   a::MatrixBlock,
#   basis_right::VectorBlock,
#   args...
#   ) 

#   block_cache = Matrix{Any}(undef,length(basis_left),length(basis_right))
#   touched = Matrix{Bool}(undef,length(basis_left),length(basis_right))
#   for i in eachindex(basis_left), j in eachindex(basis_right)
#     touched[i,j] = basis_left.touched[i] && basis_right.touched[j]
#     if touched[i,j]
#       @check a.touched[i,j]
#       block_cache[i,j] = galerkin_projection(basis_left[i],a[i,j],basis_right[j],args...)
#     end
#   end
#   return ArrayBlock(block_cache,touched)
# end

# function galerkin_projection!(
#   cache::VectorBlock,
#   basis_left::VectorBlock,
#   a::VectorBlock,
#   args...
#   )

#   for i in 1:size(cache,1)
#     if cache.touched[i]
#       @check a.touched[i]
#       galerkin_projection!(cache[i],basis_left[i],blocks(a)[i],args...)
#     end
#   end
#   return cache
# end

# function galerkin_projection!(
#   cache::MatrixBlock,
#   basis_left::VectorBlock,
#   a::MatrixBlock,
#   basis_right::VectorBlock,
#   args...
#   )

#   for i in 1:size(cache,1), j in 1:size(cache,2)
#     if cache.touched[i,j]
#       @check a.touched[i,j]
#       galerkin_projection!(cache[i,j],basis_left[i],blocks(a)[i,j],basis_right[j])
#     end
#   end
#   return cache
# end