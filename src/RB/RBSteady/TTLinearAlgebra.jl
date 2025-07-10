struct TTContraction{T,N} <: AbstractArray{T,N}
  array::Array{T,N}
end

Base.size(a::TTContraction) = size(a.array)
Base.getindex(a::TTContraction{T,N},i::Vararg{Integer,N}) where {T,N} = getindex(a.array,i...)

function contraction(::AbstractArray...)
  @abstractmethod
end

function contraction!(::Union{AbstractArray,Number}...)
  @abstractmethod
end

"""
    contraction(basis::AbstractArray,coefficient::AbstractArray) -> AbstractArray

Multiplies a reduced basis `basis` by a set of reduced coeffiecients `coefficient`.
It acts as a generalized linear combination, since `basis` might have a dimension
higher than 2.
"""
function contraction(
  basis::AbstractArray{T,3},
  coefficient::AbstractVector{S}
  ) where {T,S}

  @check size(basis,2) == length(coefficient)
  A = reshape(permutedims(basis,(1,3,2)),:,size(basis,2))
  v = A*coefficient
  s1,s2 = size(basis,1),size(basis,3)
  M = reshape(v,s1,s2)
  return M
end

function contraction!(
  cache::AbstractMatrix,
  basis::AbstractArray{T,3} where T,
  coefficient::AbstractVector,
  α::Number=1,β::Number=0
  )

  @check size(cache) == (size(basis,1),size(basis,3))
  @check size(basis,2) == length(coefficient)
  v = view(cache,:)
  A = reshape(permutedims(basis,(1,3,2)),:,size(basis,2))
  mul!(v,A,coefficient,α,β)
  return
end

function contraction!(
  cache::AbstractArray{U,3} where U,
  basis::AbstractArray{T,3} where T,
  coefficient::AbstractMatrix,
  α::Number=1,β::Number=0
  )

  @check size(cache,3) == size(coefficient,2)
  for (i,c) in enumerate(eachslice(cache,dims=3))
    @views coeff = coefficient[:,i]
    contraction!(c,basis,coeff,α,β)
  end
end

"""
    contraction(Φₗₖ::AbstractArray{T,3},Aₖ::AbstractArray{T,3}) -> AbstractArray{T,4}
    contraction(Φₗₖ::AbstractArray{T,3},Aₖ::AbstractArray{T,3},Φᵣₖ::AbstractArray{T,3}) -> AbstractArray{T,6}

Contraction of tensor train cores, as a result of a (Petrov-)Galerkin projection.
The contraction of `Aₖ` by `Φₗₖ` is the left-contraction of a TT core `Aₖ` by a
(left, test) TT core `Φₗₖ`, whereas the contraction of `Aₖ` by `Φᵣₖ` is the
right-contraction of a TT core `Aₖ` by a (right, trial) TT core `Φᵣₖ`. The dimension
of the output of a contraction involving `N` factors is: `3N - N = 2N`.
"""
Base.@propagate_inbounds function contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3}
  ) where {T,S}

  @check size(factor1,2) == size(factor2,2)
  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,3),size(factor2,1),size(factor2,3)
  ABp = permutedims(reshape(AB,s1,s2,s3,s4),(1,3,2,4))
  return TTContraction(ABp)
end

# product of cores on the component axis, for multivariate problems
Base.@propagate_inbounds function contraction(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3},
  factor3::AbstractArray{U,3}
  ) where {T,S,U}

  ncomps1 = size(factor1,2)
  ncomps3 = size(factor3,2)
  cinds = CartesianIndices((ncomps1,ncomps3))
  @check size(factor2,2) == ncomps1*ncomps3
  A = reshape(permutedims(factor1,(2,1,3)),size(factor1,2),:)
  B = reshape(permutedims(factor2,(2,1,3)),size(factor2,2),:)
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  ABC = zeros(size(A,2),size(B,2),size(C,2))
  for (iA,a) = enumerate(eachcol(A))
    for (iB,b) = enumerate(eachcol(B))
      for (iC,c) = enumerate(eachcol(C))
        for (in,n) in enumerate(cinds)
          v = @views a[n.I[1]]*b[in]*c[n.I[2]]
          _entry!(+,ABC,v,iA,iB,iC)
        end
      end
    end
  end
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(ABCp)
end

function contraction(
  factor1::AbstractArray{T,3},
  factor2::TrivialSparseCore{S},
  factor3::AbstractArray{U,3}
  ) where {T,S,U}

  factor = galerkin_projection(cores2basis(factor1),cores2basis(factor2),cores2basis(factor3))
  reshape(factor,1,1,1,size(factor)...)
end

function contraction(
  factor1::AbstractArray{T,3},
  factor2::SparseCore{S,3},
  factor3::AbstractArray{U,3}
  ) where {T,S,U}

  sparsity = factor2.sparsity
  @check size(factor1,2) == DofMaps.num_rows(sparsity)
  @check DofMaps.num_cols(sparsity) == size(factor3,2)

  A = reshape(permutedims(factor1,(1,3,2)),:,size(factor1,2))
  B = reshape(permutedims(factor2,(1,3,2)),:,size(factor2,2))
  C = reshape(permutedims(factor3,(2,1,3)),size(factor3,2),:)
  BC = _sparsemul(B,C,sparsity)
  ABC = A*BC
  s1,s2 = size(factor1,1),size(factor1,3)
  s3,s4 = size(factor2,1),size(factor2,3)
  s5,s6 = size(factor3,1),size(factor3,3)
  ABCp = permutedims(reshape(ABC,s1,s2,s3,s4,s5,s6),(1,3,5,2,4,6))
  return TTContraction(ABCp)
end

"""
    sequential_product(a::AbstractArray,b::AbstractArray...) -> AbstractArray

This function sequentially multiplies the results of several (sequential as well)
calls to `contraction`
"""
function sequential_product(::AbstractArray...)
  @abstractmethod
end

Base.@propagate_inbounds function sequential_product(
  factor1::AbstractArray{T,3},
  factor2::AbstractArray{S,3}
  ) where {T,S}

  @check size(factor1,3) == size(factor2,1)
  A = reshape(factor1,:,size(factor1,3))
  B = reshape(factor2,size(factor2,1),:)
  AB = A*B
  s1,s2,s3,s4 = size(factor1,1),size(factor1,2),size(factor2,2),size(factor2,3)
  reshape(AB,s1,s2*s3,s4)
end

Base.@propagate_inbounds function sequential_product(
  factor1::AbstractArray{T,4},
  factor2::TTContraction{S,4}
  ) where {T,S}

  @check size(factor1,1) == size(factor1,2) == 1
  @check size(factor1,3) == size(factor2,1)
  @check size(factor1,4) == size(factor2,2)

  a = vec(factor1)
  B = reshape(factor2,length(a),:)
  aB = a'*B
  s1,s2,s3,s4 = 1,1,size(factor2,3),size(factor2,4)
  reshape(aB,s1,s2,s3,s4)
end

Base.@propagate_inbounds function sequential_product(
  factor1::AbstractArray{T,6},
  factor2::TTContraction{S,6}
  ) where {T,S}

  @check size(factor1,1) == size(factor1,2) == size(factor1,3) == 1
  @check size(factor1,4) == size(factor2,1)
  @check size(factor1,5) == size(factor2,2)
  @check size(factor1,6) == size(factor2,3)

  a = vec(factor1)
  B = reshape(factor2,length(a),:)
  aB = a'*B
  s1,s2,s3,s4,s5,s6 = 1,1,1,size(factor2,4),size(factor2,5),size(factor2,6)
  reshape(aB,s1,s2,s3,s4,s5,s6)
end

function sequential_product(
  factor1::AbstractArray{T,6},
  factor2::AbstractArray{S,4}
  ) where {T,S}

  @check size(factor1,1) == size(factor1,2) == size(factor1,3) == 1
  if size(factor1,4) == size(factor2,1) && size(factor1,5) == size(factor2,2)
    _seq_prod_missing_trial(factor1,factor2)
  else size(factor1,5) == size(factor2,1) && size(factor1,6) == size(factor2,2)
    _seq_prod_missing_test(factor1,factor2)
  end
end

function sequential_product(factor1::AbstractArray,factors::AbstractArray...)
  factor2,last_factors... = factors
  sequential_product(sequential_product(factor1,factor2),last_factors...)
end

"""
    cores2basis(cores::AbstractArray{T,3}...) -> AbstractMatrix

Returns a basis in a matrix format from a list of tensor train cores `cores`. When
also supplying the indexing strategy `dof_map`, the result is reindexed accordingly
"""
function cores2basis(core::AbstractArray{T,3}) where T
  reshape(core,:,size(core,3))
end

function cores2basis(cores::AbstractArray{T,3}...) where T
  core = sequential_product(cores...)
  dropdims(core;dims=1)
end

function basis2core(basis::AbstractMatrix)
  reshape(basis,1,size(basis)...)
end

function galerkin_projection(
  cores_test::Vector{<:AbstractArray{T,3}},
  cores::Vector{<:AbstractArray{T,3}}
  ) where T

  rcores = map(contraction,cores_test,cores)
  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2))
end

function unbalanced_contractions(
  cores_test::Vector{<:AbstractArray{T,3}},
  cores::Vector{<:AbstractArray{T,3}},
  cores_trial::Vector{<:AbstractArray{T,3}}
  ) where T

  map(1:length(cores)) do d
    cond_test = isassigned(cores_test,d)
    cond_trial = isassigned(cores_trial,d)
    @notimplementedif !(cond_test || cond_trial)
    if cond_test && cond_trial
      contraction(cores_test[d],cores[d],cores_trial[d])
    elseif cond_test
      contraction(cores_test[d],cores[d])
    else
      contraction(cores[d],cores_trial[d])
    end
  end
end

function galerkin_projection(
  cores_test::Vector{<:AbstractArray{T,3}},
  cores::Vector{<:AbstractArray{T,3}},
  cores_trial::Vector{<:AbstractArray{T,3}}
  ) where T

  if length(cores_test) == length(cores) == length(cores_trial)
    rcores = map(contraction,cores_test,cores,cores_trial)
  else
    rcores = unbalanced_contractions(cores_test,cores,cores_trial)
  end

  rcore = sequential_product(rcores...)
  dropdims(rcore;dims=(1,2,3))
end

# supremizer computation for tensor train decompositions

function tt_supremizer(
  X::Vector{<:Factorization},
  B::Vector{<:AbstractSparseMatrix},
  cores_d::Vector{<:AbstractArray{T,3}}
  ) where T

  @check length(X) == length(B)
  @check length(X) ≤ length(cores_d)
  n = length(X)
  supr_cores = Vector{Array{T,3}}(undef,n)
  for d in 1:n
    cur_core = cores_d[d]
    XinvB = X[d] \ B[d]
    supr_cores[d] = _sparse_rescaling(XinvB,cur_core)
  end
  return supr_cores
end

function tt_supremizer(
  X::Vector{<:Factorization},
  B::GenericRankTensor,
  cores_d::Vector{<:AbstractArray{T,3}}
  ) where T

  nB = length(get_decomposition(B))
  vec_supr = Vector{Vector{Array{T,3}}}(undef,nB)
  for iB in 1:nB
    Bi = get_decomposition(B)[iB]
    Bfactors = get_factors(Bi)
    vec_supr[iB] = tt_supremizer(X,Bfactors,cores_d)
  end
  supr_cores = _block_cores_add_component(vec_supr)
  if length(cores_d) > length(X)
    push!(supr_cores,last(cores_d))
  end
  return supr_cores
end

# utils

Base.@propagate_inbounds function _sparsemul(B,C,sparsity::SparsityCSC)
  BC = zeros(size(B,1),DofMaps.num_rows(sparsity),size(C,2))
  rv = rowvals(sparsity)
  for (iB,b) in enumerate(eachrow(B))
    for (iC,c) in enumerate(eachcol(C))
      for (irow,ci) in enumerate(c)
        for nzi in nzrange(sparsity,irow)
          _entry!(+,BC,b[nzi]*ci,iB,rv[nzi],iC)
        end
      end
    end
  end
  return reshape(permutedims(BC,(2,1,3)),size(BC,2),:)
end

@inline function _entry!(combine::Function,A::AbstractArray{T,3},v,i,j,k) where T
  aijk = A[i,j,k]
  A[i,j,k] = combine(aijk,v)
  A
end

Base.@propagate_inbounds function _seq_prod_missing_trial(
  factor1::AbstractArray{T,6},
  factor2::AbstractArray{S,4}
  ) where {T,S}

  factor1′ = permutedims(factor1,(1,2,3,6,4,5))
  factor12′ = _seq_prod_missing_test(factor1′,factor2)
  permutedims(factor12′,(1,2,3,5,6,4))
end

Base.@propagate_inbounds function _seq_prod_missing_test(
  factor1::AbstractArray{T,6},
  factor2::AbstractArray{S,4}
  ) where {T,S}

  n = size(factor1,5)*size(factor1,6)
  A = reshape(factor1,:,n)
  B = reshape(factor2,n,:)
  AB = A*B
  s1,s2,s3,s4,s5,s6 = 1,1,1,size(factor1,4),size(factor2,3),size(factor2,4)
  reshape(AB,s1,s2,s3,s4,s5,s6)
end

# rescale a 3d-core by a (sparse) matrix

function _sparse_rescaling(X::GenericRankTensor,cores::Vector{<:AbstractArray{T,3}}) where T
  sum(map(k -> _sparse_rescaling(get_decomposition(X,k),cores),1:rank(X)))
end

function _sparse_rescaling(X::Rank1Tensor,cores::Vector{<:AbstractArray{T,3}}) where T
  map(_sparse_rescaling,get_factors(X),cores)
end

function _sparse_rescaling(X::AbstractSparseMatrix,core::AbstractArray{T,3}) where T
  prev_rank = size(core,1)
  cur_size = size(core,2)
  cur_size′ = size(X,1)
  M = reshape(core,prev_rank*cur_size,:)

  X′ = kron(X,I(prev_rank))
  XM = X′*M
  Xcore = reshape(XM,prev_rank,cur_size′,:)

  return Xcore
end

# empirical interpolation

function basis_index(cores_indices::Table,i::Integer,l::Integer)
  @assert l > 0
  l == 1 && return i
  lprev = l-1
  pinilprev = cores_indices.ptrs[lprev]
  pendlprev = cores_indices.ptrs[lprev+1]
  rankprev = pendlprev-pinilprev
  icurr = slow_index(i,rankprev)
  iprev = cores_indices.data[pinilprev-1+fast_index(i,rankprev)]
  iprevs = basis_index(cores_indices,iprev,lprev)
  return (iprevs...,icurr)
end

function _get_indices(
  ::Val{1},
  cores_indices::Table,
  dof_map::AbstractArray{Ti,D}
  )::Vector{Ti} where {Ti,D}

  L = length(cores_indices)
  piniL = cores_indices.ptrs[L]
  pendL = cores_indices.ptrs[L+1]-1
  space_indices = zeros(Ti,pendL-piniL+1)
  for (k,pk) in enumerate(piniL:pendL)
    ik = cores_indices.data[pk]
    indices_space_k = basis_index(cores_indices,ik,L)
    index_space_k = dof_map[CartesianIndex(indices_space_k)]
    space_indices[k] = index_space_k
  end

  return space_indices
end

function _get_indices(
  ::Val{2},
  cores_indices::Table,
  dof_map::AbstractArray{Ti,D}
  )::NTuple{2,Vector{Ti}} where {Ti,D}

  L = length(cores_indices)
  piniL = cores_indices.ptrs[L]
  pendL = cores_indices.ptrs[L+1]-1
  space_indices = zeros(Ti,pendL-piniL+1)
  time_indices = zeros(Ti,pendL-piniL+1)
  for (k,pk) in enumerate(piniL:pendL)
    ik = cores_indices.data[pk]
    indices_space_k...,index_time_k = basis_index(cores_indices,ik,L)
    index_space_k = dof_map[CartesianIndex(indices_space_k)]
    space_indices[k] = index_space_k
    time_indices[k] = index_time_k
  end

  return space_indices,time_indices
end

get_basis_indices(k,cores_indices,dof_map) = _get_indices(k,cores_indices,dof_map)

for T in (:TrivialSparseMatrixDofMap,:SparseMatrixDofMap)
  @eval begin
    function get_basis_indices(k::Val{1},cores_indices::Table,dof_map::$T)
      space_indices = _get_indices(k,cores_indices,dof_map)
      recast_split_indices(space_indices,dof_map)
    end

    function get_basis_indices(k::Val{2},cores_indices::Table,dof_map::$T)
      space_indices,time_indices = _get_indices(k,cores_indices,dof_map)
      space_indices′ = recast_split_indices(space_indices,dof_map)
      return space_indices′,time_indices
    end
  end
end

function get_basis_indices(cores_indices::Table,dof_map::AbstractArray{Ti,D}) where {Ti,D}
  L = length(cores_indices)
  N = L - D + 1
  @check N ∈ (1,2)
  get_basis_indices(Val{N}(),cores_indices,dof_map)
end
