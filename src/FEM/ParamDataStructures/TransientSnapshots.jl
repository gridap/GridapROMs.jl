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

function Snapshots(s::AbstractParamVector,i::AbstractDofMap,r::TransientRealisation)
  data = get_all_data(s)
  data′ = reshape(data,:,num_params(r),num_times(r))
  s′ = ConsecutiveParamArray(data′)
  Snapshots(s′,i,r)
end

function Snapshots(s::AbstractParamMatrix,i::TrivialDofMap,r::TransientRealisation)
  Snapshots(get_all_data(s),i,r)
end

function get_param_data(s::TransientSnapshots)
  data = get_all_data(s)
  ncols = num_times(s)*num_params(s)
  ConsecutiveParamArray(reshape(data,:,ncols))
end

get_initial_data(s::TransientSnapshots) = @abstractmethod
get_initial_param_data(s::TransientSnapshots) = @abstractmethod

function _select_snapshots(s::TransientSnapshots,pindex)
  prange = _format_index(pindex)
  trange = 1:num_times(s)
  drange = view(get_all_data(s),:,prange,trange)
  rrange = get_realisation(s)[prange,trange]
  Snapshots(drange,get_dof_map(s),rrange)
end

function param_getindex(
  s::TransientSnapshots{T,N},
  pindex::Integer,
  tindex::Integer
  ) where {T,N}
  view(get_all_data(s),_ncolons(Val{N-2}())...,pindex,tindex)
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
  s::AbstractParamMatrix,
  s0::Tuple{Vararg{AbstractParamVector}},
  i::AbstractDofMap,
  r::TransientRealisation
  )
  initial_data = get_all_data.(s0)
  snaps = Snapshots(s,i,r)
  TransientSnapshotsWithIC(initial_data,snaps)
end

function Snapshots(
  s::AbstractParamVector,
  s0::Tuple{Vararg{AbstractParamVector}},
  i::AbstractDofMap,
  r::TransientRealisation
  )
  data = get_all_data(s)
  data′ = reshape(data,:,num_params(r),num_times(r))
  s′ = ConsecutiveParamArray(data′)
  Snapshots(s′,s0,i,r)
end

get_all_data(s::TransientSnapshotsWithIC) = get_all_data(s.snaps)
get_initial_data(s::TransientSnapshotsWithIC) = s.initial_data
get_initial_param_data(s::TransientSnapshotsWithIC) = ConsecutiveParamArray.(s.initial_data)
DofMaps.get_dof_map(s::TransientSnapshotsWithIC) = get_dof_map(s.snaps)
get_realisation(s::TransientSnapshotsWithIC) = get_realisation(s.snaps)

function Base.getindex(s::TransientSnapshotsWithIC{T,N},i::Vararg{Integer,N}) where {T,N}
  getindex(s.snaps,i...)
end

function Base.setindex!(s::TransientSnapshotsWithIC{T,N},v,i::Vararg{Integer,N}) where {T,N}
  setindex!(s.snaps,v,i...)
end

function _select_snapshots(s::TransientSnapshotsWithIC,pindex)
  prange = _format_index(pindex)
  d0range = map(d0 -> view(d0,:,prange),s.initial_data)
  srange = _select_snapshots(s.snaps,pindex)
  TransientSnapshotsWithIC(d0range,srange)
end

function param_cat(v::AbstractVector{<:TransientSnapshotsWithIC})
  _get_snaps(s) = s.snaps
  param_cat(map(_get_snaps,v))
end

const TransientReshapedSnapshots{T,N,I,R<:TransientRealisation,A,B} = ReshapedSnapshots{T,N,I,R,A,B}

function get_param_data(s::TransientReshapedSnapshots)
  data = get_all_data(s)
  ncols = num_times(s)*num_params(s)
  ConsecutiveParamArray(reshape(data,:,ncols))
end

function Snapshots(s::AbstractParamMatrix,i::AbstractDofMap,r::TransientRealisation)
  data = get_all_data(s)
  param_data = s
  dims = (size(i)...,num_params(r),num_times(r))
  idata = reshape(data,dims)
  ReshapedSnapshots(idata,param_data,i,r)
end

function Snapshots(
  s::ParamSparseMatrix,
  i::TrivialSparseMatrixDofMap,
  r::TransientRealisation
  )
  T = eltype2(s)
  data = get_all_data(s)
  data′ = reshape(data,:,num_params(r),num_times(r))
  param_data = s
  ReshapedSnapshots(data′,param_data,i,r)
end

function Snapshots(s::ParamSparseMatrix,i::SparseMatrixDofMap,r::TransientRealisation)
  T = eltype2(s)
  data = get_all_data(s)
  param_data = s
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
  ReshapedSnapshots(idata,param_data,i,r)
end

function _select_snapshots(s::TransientReshapedSnapshots{T,N},pindex) where {T,N}
  np = num_params(s)
  prange = _format_index(pindex)
  trange = 1:num_times(s)
  drange = view(get_all_data(s),_ncolons(Val{N-2}())...,prange,trange)
  pdrange = _get_param_data(s.param_data,prange,trange;nparams=np)
  rrange = get_realisation(s)[prange,trange]
  ReshapedSnapshots(drange,pdrange,get_dof_map(s),rrange)
end

function _get_param_data(pdata::ConsecutiveParamMatrix,prange,trange;kwargs...)
  ConsecutiveParamArray(view(pdata.data,:,prange,trange))
end

# in practice, when dealing with the Jacobian, the param data is never fetched
function _get_param_data(
  pdata::ConsecutiveParamSparseMatrixCSC,prange,trange;
  nparams=Int(param_length(pdata)/length(trange))
  )
  datarange = view(pdata.data,:,range_1d(prange,trange,nparams))
  ConsecutiveParamSparseMatrixCSC(pdata.m,pdata.n,pdata.colptr,pdata.rowval,datarange)
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
DofMaps.get_dof_map(s::ModeTransientSnapshots) = s.dof_map
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