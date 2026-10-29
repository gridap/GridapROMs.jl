"""
    const TransientSnapshots{T,N,I,R<:TransientRealisation} = Snapshots{T,N,I,R}

Transient specialization of a `Snapshots`

Subtypes:
- [`TransientSnapshotsWithIC`](@ref)
"""
const TransientSnapshots{T,N,I,R<:TransientRealisation} = Snapshots{T,N,I,R}

num_times(s::TransientSnapshots) = num_times(get_realisation(s))

"""
    get_initial_param_data(s::TransientSnapshots) -> AbstractParamArray

Initial conditions of the transient snapshots `s`
"""
get_initial_param_data(s::TransientSnapshots) = @abstractmethod

function flatten(s::TransientSnapshots)
  d = get_all_data(s)
  reshape(d,:,num_params(s),num_times(s))
end

"""
    struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R}} <: TransientSnapshots{T,N,I,R}
      initial_param_data::A
      snaps::B
    end

Stores a [`TransientSnapshots`](@ref) `snaps` alongside a parametric initial condition `initial_param_data`
"""
struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R}} <: TransientSnapshots{T,N,I,R}
  initial_param_data::A
  snaps::B
end

function Snapshots(s::AbstractParamArray,s0::Tuple,r::AbstractRealisation)
  i = VectorDofMap(innerlength(s))
  Snapshots(s,s0,i,r)
end

function Snapshots(
  s::AbstractParamArray,
  s0::Tuple{Vararg{AbstractParamVector}},
  i::AbstractDofMap,
  r::TransientRealisation
  )
  
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(s0,snaps)
end

get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_param_data(s::TransientSnapshotsWithIC) = get_param_data(s.snaps)
get_initial_param_data(s::TransientSnapshotsWithIC) = s.initial_param_data
get_dof_map(s::TransientSnapshotsWithIC) = get_dof_map(s.snaps)
get_realisation(s::TransientSnapshotsWithIC) = get_realisation(s.snaps)

function Base.getindex(s::TransientSnapshotsWithIC{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(s.snaps,i...)
end

function Base.setindex!(s::TransientSnapshotsWithIC{T,N},v,i::Vararg{Integer,N}) where {T,N}
  setindex!(s.snaps,v,i...)
end

function select_snapshots(s::TransientSnapshotsWithIC,pindex)
  prange = _format_index(pindex)
  d0range = map(d0 -> select_param_data(d0,prange),s.initial_param_data)
  srange = select_snapshots(s.snaps,pindex)
  TransientSnapshotsWithIC(d0range,srange)
end

"""
    select_times(s::TransientSnapshots,tindex) -> TransientSnapshots

Restriction of `s` to the time steps of index `tindex`
"""
function select_times(s::TransientSnapshotsWithIC,tindex)
  srange = select_times(s.snaps,tindex)
  TransientSnapshotsWithIC(s.initial_param_data,srange)
end

function param_cat(v::AbstractVector{<:TransientSnapshotsWithIC})
  param_cat(map(s -> s.snaps,v))
end

"""
    const TransientGenericSnapshots{T,N,I,R<:TransientRealisation,A,B} = GenericSnapshots{T,N,I,R,A,B}
"""
const TransientGenericSnapshots{T,N,I,R<:TransientRealisation,A,B} = GenericSnapshots{T,N,I,R,A,B}

function select_snapshots(s::TransientGenericSnapshots,pindex)
  nparams = num_params(s)
  prange = _format_index(pindex)
  trange = 1:num_times(s)
  tprange = range_1d(prange,trange,nparams)
  GenericSnapshots(
    select_all_data(s.data,tprange),
    select_param_data(s.param_data,tprange),
    get_dof_map(s),
    get_realisation(s)[prange,trange]
  )
end

function select_times(s::TransientGenericSnapshots,tindex)
  nparams = num_params(s)
  prange = 1:nparams
  trange = _format_index(tindex)
  tprange = range_1d(prange,trange,nparams)
  GenericSnapshots(
    select_all_data(s.data,tprange),
    select_param_data(s.param_data,tprange),
    get_dof_map(s),
    get_realisation(s)[prange,trange]
  )
end

"""
    const TransientSparseSnapshots{T,N,I<:AbstractSparseDofMap,R<:TransientRealisation} = TransientSnapshots{T,N,I,R}

Transient version of [`SparseSnapshots`](@ref)
"""
const TransientSparseSnapshots{T,N,I<:AbstractSparseDofMap,R<:TransientRealisation} = TransientSnapshots{T,N,I,R}

# block snapshots

"""
    const TransientBlockSnapshots{S<:TransientSnapshots,N} = BlockSnapshots{S,N}
"""
const TransientBlockSnapshots{S<:TransientSnapshots,N} = BlockSnapshots{S,N}

num_times(s::TransientBlockSnapshots) = num_times(get_realisation(s))
function get_initial_param_data(s::TransientBlockSnapshots)
  x = first(blocks(s))
  data0 = ()
  for i in eachindex(get_initial_param_data(x))
    d0i = map(x -> get_initial_param_data(x)[i],blocks(s))
    data0 = (data0...,mortar(d0i))
  end
  return data0
end

function select_snapshots(s::TransientBlockSnapshots,pindex)
  array = map(sj -> select_snapshots(sj,pindex),blocks(s))
  return BlockSnapshots(array)
end

function select_times(s::TransientBlockSnapshots,tindex)
  array = map(sj -> select_times(sj,tindex),blocks(s))
  return BlockSnapshots(array)
end

function Snapshots(
  data::BlockParamArray,
  data0::Tuple{Vararg{BlockParamArray}},
  i::AbstractArray{<:AbstractDofMap},
  r::TransientRealisation
  )

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)
  array = map(enumerate(block_values)) do (j,dataj)
    data0j = map(d0 -> blocks(d0)[j],data0)
    Snapshots(dataj,data0j,i[j],r)
  end
  BlockSnapshots(array)
end

function Snapshots(
  data::AbstractParamArray,
  data0::Tuple{Vararg{AbstractParamArray}},
  i::AbstractArray{<:AbstractDofMap},
  r::TransientRealisation
  )

  ids = offset_indices(i)
  array = map(enumerate(i)) do (j,ij)
    dataj = get_param_entry(data,ids[j]...)
    data0j = map(d0 -> get_param_entry(d0,ids[j]...),data0)
    Snapshots(dataj,data0j,ij,r)
  end
  BlockSnapshots(array)
end

# utils

function Snapshots(
  a::ArrayContributionTuple,
  i::ArrayContributionTuple,
  r::TransientRealisation
  )

  ContributionTuple(map((ai,ii)->Snapshots(ai,ii,r),a.tuple,i.tuple))
end

function select_snapshots(a::ArrayContributionTuple,pindex)
  map(ai->select_snapshots(ai,pindex),a)
end

function select_times(a::ArrayContributionTuple,tindex)
  map(ai->select_times(ai,tindex),a)
end

"""
    change_mode(a::AbstractMatrix,np::Integer) -> AbstractMatrix

Converts the space-mode unfolding `a` of transient snapshots with `np` parameters
into their time-mode unfolding (time steps × (space DOFs, parameters))
"""
function change_mode(a::AbstractMatrix,np::Integer)
  n1 = size(a,1)
  n2 = Int(size(a,2)/np)
  a′ = zeros(eltype(a),n2,n1*np)
  @inbounds for i = 1:np
    @views a′[:,(i-1)*n1+1:i*n1] = a[:,i:np:np*n2]'
  end
  return a′
end

# utils

function unfold(A::SubArray{T,N},dims::Dims) where {T,N}
  @check length(A) == prod(dims)
  nsplit = length(dims)-N+1
  nsplit == 1 && return A
  @check nsplit == 2 "unfold only supports splitting the trailing dimension into (nparams,ntimes)"
  leading = A.indices[1:end-1]
  lind = A.indices[end]
  r2d = lind.parent
  parent = A.parent
  leading_dims = size(parent)[1:end-1]
  ncols = size(parent,ndims(parent))
  np = r2d.scale
  nt = ncols ÷ np
  parent′ = reshape(parent,leading_dims...,np,nt)
  view(parent′,leading...,r2d.axis1,r2d.axis2)
end