"""
    const TransientSnapshots{T,N,I,R<:TransientRealisation} = Snapshots{T,N,I,R}

Transient specialization of a `Snapshots`

Subtypes:
- [`TransientSnapshotsWithIC`](@ref)
- [`ModeTransientSnapshots`](@ref)
"""
const TransientSnapshots{T,N,I,R<:TransientRealisation} = Snapshots{T,N,I,R}

space_dofs(s::TransientSnapshots{T,N}) where {T,N} = size(get_all_data(s))[1:N-2]

num_times(s::TransientSnapshots) = num_times(get_realisation(s))

Base.size(s::TransientSnapshots) = (space_dofs(s)...,num_params(s),num_times(s))

function Snapshots(s::AbstractParamArray,i::TrivialDofMap,r::TransientRealisation)
  dims = (length(i),num_params(r),num_times(r))
  idata = reshape(get_all_data(s),dims)
  GenericSnapshots(idata,s,i,r)
end

function Snapshots(s::AbstractParamArray,i::AbstractDofMap,r::TransientRealisation)
  data = get_all_data(s)
  param_data = s
  if _is_one_to(i)
    dims = (size(i)...,num_params(r),num_times(r))
    idata = reshape(data,dims)
    return GenericSnapshots(idata,param_data,i,r)
  end
  T = eltype2(s)
  idata = zeros(T,size(i)...,num_params(r),num_times(r))
  for it in 1:num_times(r), ip in 1:num_params(r)
    ipt = (it-1)*num_params(r)+ip
    for k in CartesianIndices(i)
      k′ = i[k]
      if k′ > 0
        idata[k.I...,ip,it] = data[k′,ipt]
      end
    end
  end
  GenericSnapshots(idata,param_data,i,r)
end

get_initial_data(s::TransientSnapshots) = @abstractmethod
get_initial_param_data(s::TransientSnapshots) = @abstractmethod

function param_getindex(
  s::TransientSnapshots{T,N},
  pindex::Integer,
  tindex::Integer
  ) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-2}())...,pindex,tindex)
end

function flatten(s::TransientSnapshots)
  d = get_all_data(s)
  reshape(d,:,num_params(s),num_times(s))
end

"""
    struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R}} <: TransientSnapshots{T,N,I,R}
      initial_data::A
      snaps::B
    end

Stores a [`TransientSnapshots`](@ref) `snaps` alongside a parametric initial condition `initial_data`
"""
struct TransientSnapshotsWithIC{T,N,I,R,A,B<:TransientSnapshots{T,N,I,R}} <: TransientSnapshots{T,N,I,R}
  initial_data::A
  snaps::B
end

function Snapshots(
  s::AbstractParamArray,
  s0::Tuple{Vararg{AbstractParamVector}},
  i::AbstractDofMap,
  r::TransientRealisation
  )
  initial_data = get_all_data.(s0)
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(initial_data,snaps)
end

get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_param_data(s::TransientSnapshotsWithIC) = get_param_data(s.snaps)
get_initial_data(s::TransientSnapshotsWithIC) = s.initial_data
get_initial_param_data(s::TransientSnapshotsWithIC) = ConsecutiveParamArray.(s.initial_data)
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
  d0range = map(d0 -> view(d0,:,prange),s.initial_data)
  srange = select_snapshots(s.snaps,pindex)
  TransientSnapshotsWithIC(d0range,srange)
end

function select_times(s::TransientSnapshotsWithIC,tindex)
  srange = select_times(s.snaps,tindex)
  TransientSnapshotsWithIC(s.initial_data,srange)
end

function param_cat(v::AbstractVector{<:TransientSnapshotsWithIC})
  _get_snaps(s) = s.snaps
  param_cat(map(_get_snaps,v))
end

function change_dof_map(s::TransientSnapshotsWithIC,i)
  TransientSnapshotsWithIC(s.initial_data,change_dof_map(s.snaps,i))
end

const TransientGenericSnapshots{T,N,I,R<:TransientRealisation,A,B} = GenericSnapshots{T,N,I,R,A,B}

function select_snapshots(s::TransientGenericSnapshots{T,N},pindex) where {T,N}
  np = num_params(s)
  prange = _format_index(pindex)
  trange = 1:num_times(s)
  drange = view(get_all_data(s),_ncolons(Val{N-2}())...,prange,trange)
  pdrange = _get_param_data(s.param_data,prange,trange;nparams=np)
  rrange = get_realisation(s)[prange,trange]
  GenericSnapshots(drange,pdrange,get_dof_map(s),rrange)
end

function select_times(s::TransientGenericSnapshots{T,N},tindex) where {T,N}
  np = num_params(s)
  prange = 1:np
  trange = _format_index(tindex)
  drange = view(get_all_data(s),_ncolons(Val{N-2}())...,prange,trange)
  pdrange = _get_param_data(s.param_data,prange,trange;nparams=np)
  rrange = get_realisation(s)[prange,trange]
  GenericSnapshots(drange,pdrange,get_dof_map(s),rrange)
end

"""
"""
const TransientSparseSnapshots{T,N,I<:AbstractSparseDofMap,R<:TransientRealisation} = Snapshots{T,N,I,R}

# block snapshots

function Snapshots(
  data::BlockParamArray{T,N},
  data0::Tuple{Vararg{BlockParamArray}},
  i::ArrayBlock{<:AbstractDofMap},
  r::TransientRealisation
  ) where {T,N}

  block_values = blocks(data)
  s = size(block_values)
  @check s == size(i)

  array = Array{Any,N}(undef,s)
  for j in eachindex(block_values)
    dataj = block_values[j]
    data0j = map(d0 -> blocks(d0)[j],data0)
    if i.touched[j]
      array[j] = Snapshots(dataj,data0j,i[j],r)
    end
  end

  BlockSnapshots(array,i.touched)
end

for f in (:get_initial_data,:get_initial_param_data)
  @eval begin
    function $f(s::BlockSnapshots{N}) where N
      t = $f(testitem(s))
      a = ()
      for (j,tj) in enumerate(t)
        vj = Array{typeof(tj),N}(undef,size(s))
        for i in eachindex(s.touched)
          if s.touched[i]
            vj[i] = $f(s[i])[j]
          end
        end
        a = (a...,mortar(vj))
      end
      return a
    end
  end
end

function select_times(s::BlockSnapshots{N},tindex) where N
  array = Array{Any,N}(undef,size(s))
  for i in eachindex(s.touched)
    if s.touched[i]
      array[i] = select_times(s[i],tindex)
    end
  end
  return BlockSnapshots(array,s.touched)
end

# mode snapshots

abstract type ModeAxes end
struct Mode1Axes <: ModeAxes end
struct Mode2Axes <: ModeAxes end

struct ModeTransientSnapshots{M<:ModeAxes,T,I,R,A<:AbstractMatrix{T}} <: TransientSnapshots{T,2,I,R}
  mode::M
  data::A
  dof_map::I
  realisation::R
end

function ModeTransientSnapshots(data,i,r)
  ModeTransientSnapshots(Mode1Axes(),data,i,r)
end

Base.size(s::ModeTransientSnapshots) = size(s.data)

get_all_data(s::ModeTransientSnapshots) = s.data
get_dof_map(s::ModeTransientSnapshots) = s.dof_map
get_realisation(s::ModeTransientSnapshots) = s.realisation

function Base.getindex(s::ModeTransientSnapshots,i,j)
  getindex(s.data,i,j)
end

function Base.setindex!(s::ModeTransientSnapshots,v,i,j)
  setindex!(s.data,v,i,j)
end

function get_mode1(s::TransientSnapshots)
  ns = num_space_dofs(s)
  data = get_all_data(s)
  m1 = reshape(data,ns,:)
  i = get_dof_map(s)
  r = get_realisation(s)
  ModeTransientSnapshots(m1,i,r)
end

function get_mode2(s::TransientSnapshots)
  mode1 = get_mode1(s)
  m2 = change_mode(mode1.data,num_params(s))
  ModeTransientSnapshots(Mode2Axes(),m2,get_dof_map(s),get_realisation(s))
end

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

function Snapshots(
  a::TupOfArrayContribution,
  i::TupOfArrayContribution,
  r::TransientRealisation
  )

  map((a,i)->Snapshots(a,i,r),a,i)
end

function select_snapshots(a::TupOfArrayContribution,pindex)
  map(a->select_snapshots(a,pindex),a)
end

function select_times(a::TupOfArrayContribution,tindex)
  map(a->select_times(a,tindex),a)
end

function change_dof_map(a::TupOfArrayContribution,i::TupOfArrayContribution)
  a′ = ()
  for j in eachindex(a)
    a′ = (a′...,change_dof_map(a[j],i[j]))
  end
  return a′
end

function _get_param_data(
  pdata::ConsecutiveParamArray{T,N},
  prange,trange;
  nparams=Int(param_length(pdata)/length(trange))
  ) where {T,N}

  datarange = view(pdata.data,_ncolons(Val{N}())...,range_1d(prange,trange,nparams))
  ConsecutiveParamArray(datarange)
end

# in practice, when dealing with the Jacobian, the param data is never fetched
function _get_param_data(
  pdata::ConsecutiveParamSparseMatrixCSC,prange,trange;
  nparams=Int(param_length(pdata)/length(trange))
  )

  datarange = view(pdata.data,:,range_1d(prange,trange,nparams))
  ConsecutiveParamSparseMatrixCSC(pdata.m,pdata.n,pdata.colptr,pdata.rowval,datarange)
end