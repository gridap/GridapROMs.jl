function PartitionedArrays.allocate_gather_impl(
  snd,
  destination,
  ::Type{T}
  ) where T<:AbstractParamVector

  l = map(innerlength,snd)
  l_dest = gather(l;destination)
  S = eltype2(T)
  function f(l,snd)
    ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
    ndata = ptrs[end]-1
    data = Vector{S}(undef,ndata)
    plength = param_length(snd)
    pdata = parameterise(data,plength)
    ParamJaggedArray{S,Int32}(pdata,ptrs)
  end
  if isa(destination,Integer)
    function g(l,snd)
      ptrs = Vector{Int32}(undef,1)
      data = Vector{S}(undef,0)
      plength = param_length(snd)
      pdata = parameterise(data,plength)
      ParamJaggedArray(pdata,ptrs)
    end
    rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
  else
    @assert destination === :all
    rcv = map(f,l_dest,snd)
  end
  rcv
end

function PartitionedArrays.allocate_scatter_impl(
  snd,
  source,
  ::Type{T}
  ) where T<:AbstractParamVector

  S = eltype2(T)
  counts,plength = map(snd) do snd
    innerlength(snd),param_length(snd)
  end |> tuple_of_arrays
  counts_scat = scatter(counts;source)
  map(counts_scat,plength) do count,plength
    data = Vector{S}(undef,count)
    parameterise(data,plength)
  end
end

function PartitionedArrays.assemble_impl!(
  f,
  vector_partition,
  cache,
  ::Type{<:ParamVectorAssemblyCache}
  )

  buffer_snd = map(vector_partition,cache) do values,cache
    local_indices_snd = cache.local_indices_snd
    for (p,lid) in enumerate(local_indices_snd.data)
      for k in _param_eachindex(values)
        cache.buffer_snd.data[p,k] = _getindex(values,lid,k)
      end
    end
    cache.buffer_snd
  end
  neighbors_snd,neighbors_rcv,buffer_rcv = map(cache) do cache
    cache.neighbors_snd,cache.neighbors_rcv,cache.buffer_rcv
  end |> tuple_of_arrays
  graph = ExchangeGraph(neighbors_snd,neighbors_rcv)
  t = exchange!(buffer_rcv,buffer_snd,graph)
  # Fill values from rcv buffer asynchronously
  @async begin
    wait(t)
    map(vector_partition,cache) do values,cache
      local_indices_rcv = cache.local_indices_rcv
      for (p,lid) in enumerate(local_indices_rcv.data)
        for k in _param_eachindex(values)
          v = f(_getindex(values,lid,k),cache.buffer_rcv.data[p,k])
          _setindex!(values,v,lid,k)
        end
      end
    end
    nothing
  end
end

function PartitionedArrays.assemble_impl!(
  f,
  matrix_partition,
  cache,
  ::Type{<:ParamSparseMatrixAssemblyCache}
  )

  vcache = map(i->i.cache,cache)
  data = map(nonzeros,matrix_partition)
  assemble!(f,data,vcache)
end

function PartitionedArrays.assemble_impl!(
  f,
  vector_partition,
  cache,
  ::Type{<:ParamJaggedArrayAssemblyCache}
  )

  vcache = map(i->i.cache,cache)
  data = map(PartitionedArrays.getdata,vector_partition)
  assemble!(f,data,vcache)
end

function PartitionedArrays.allocate_exchange_impl(
  snd,
  graph,
  ::Type{T}
  ) where T<:AbstractParamVector

  S = eltype2(T)
  n_snd,plength = map(snd) do snd
    innerlength(snd),param_length(snd)
  end |> tuple_of_arrays
  n_rcv = PartitionedArrays.exchange_fetch(n_snd,graph)
  rcv = map(n_rcv,plength) do n_rcv,plength
    ptrs = zeros(Int32,length(n_rcv)+1)
    ptrs[2:end] = n_rcv
    length_to_ptrs!(ptrs)
    n_data = ptrs[end]-1
    data = Vector{S}(undef,n_data)
    pdata = parameterise(data,plength)
    ParamJaggedArray(pdata,ptrs)
  end
  rcv
end

#

function PartitionedArrays.allocate_gather_impl(
  snd,
  destination,
  ::Type{T}
  ) where T<:AbstractMatrix

  l = map(_innersize,snd)
  l_dest = gather(l;destination)
  S = eltype(T)
  function f(l,snd)
    ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
    ndata = ptrs[end]-1
    plength = _get_plength(snd)
    pdata = Matrix{S}(undef,ndata,plength)
    ParamJaggedArray{S,Int32}(pdata,ptrs)
  end
  if isa(destination,Integer)
    function g(l,snd)
      ptrs = Vector{Int32}(undef,1)
      plength = _get_plength(snd)
      pdata = Matrix{S}(undef,0,plength)
      ParamJaggedArray(pdata,ptrs)
    end
    rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
  else
    @assert destination === :all
    rcv = map(f,l_dest,snd)
  end
  rcv
end

function PartitionedArrays.allocate_scatter_impl(
  snd,
  source,
  ::Type{T}
  ) where T<:AbstractMatrix

  S = eltype(T)
  counts,plength = map(snd) do snd
    size(snd,1),_get_plength(snd)
  end |> tuple_of_arrays
  counts_scat = scatter(counts;source)
  map(counts_scat,plength) do count,plength
    Matrix{S}(undef,count,plength)
  end
end

function PartitionedArrays.allocate_exchange_impl(
  snd,
  graph,
  ::Type{T}
  ) where T<:AbstractMatrix

  S = eltype(T)
  n_snd,plength = map(snd) do snd
    map(x->size(x,1),snd),_get_plength(snd)
  end |> tuple_of_arrays
  n_rcv = PartitionedArrays.exchange_fetch(n_snd,graph)
  rcv = map(n_rcv,plength) do n_rcv,plength
    ptrs = zeros(Int32,length(n_rcv)+1)
    ptrs[2:end] = n_rcv
    length_to_ptrs!(ptrs)
    n_data = ptrs[end]-1
    pdata = Matrix{S}(undef,n_data,plength)
    ParamJaggedArray(pdata,ptrs)
  end
  rcv
end

for T in (:AbstractMatrix,:ConsecutiveParamVector,:ParamJaggedArray)
  @eval begin
    function PartitionedArrays.gather_impl!(
      rcv::MPIArray,
      snd::MPIArray,
      destination,
      ::Type{<:$T}
      )

      @assert rcv.comm === snd.comm
      @assert isa(rcv.item,ParamJaggedArray)
      comm = snd.comm
      sdata = _data(snd.item)
      Tv = eltype(sdata)
      plength = size(sdata,2)
      rdata = rcv.item.data
      rptrs = rcv.item.ptrs
      counts = PartitionedArrays.ptrs_to_counts(rptrs)
      if isa(destination,Integer)
        root = destination-1
        if MPI.Comm_rank(comm) == root
          @assert length(rcv.item) == MPI.Comm_size(comm)
          rdata[rptrs[destination]:rptrs[destination+1]-1,:] .= sdata
          for k in 1:plength
            MPI.Gatherv!(MPI.IN_PLACE,MPI.VBuffer(view(rdata,:,k),counts),root,comm)
          end
        else
          for k in 1:plength
            MPI.Gatherv!(convert(Vector{Tv},view(sdata,:,k)),nothing,root,comm)
          end
        end
      else
        @assert destination === :all
        @assert length(rcv.item) == MPI.Comm_size(comm)
        for k in 1:plength
          sk = convert(Vector{Tv},view(sdata,:,k))
          MPI.Allgatherv!(sk,MPI.VBuffer(view(rdata,:,k),counts),comm)
        end
      end
      rcv
    end

    function PartitionedArrays.exchange_impl!(rcv,snd,graph,::Type{<:$T})
      @assert PartitionedArrays.is_consistent(graph)
      @assert eltype(rcv) <: ParamJaggedArray
      snd_ids = graph.snd
      rcv_ids = graph.rcv
      @assert length(rcv_ids) == length(rcv)
      @assert length(rcv_ids) == length(snd)
      for rcv_id in eachindex(rcv_ids)
        for (i,snd_id) in enumerate(rcv_ids[rcv_id])
          snd_snd_id = JaggedArray(snd[snd_id])
          j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
          ptrs_rcv = rcv[rcv_id].ptrs
          ptrs_snd = snd_snd_id.ptrs
          plength = _get_plength(snd[snd_id])
          @assert ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
          for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
            p_rcv = p+ptrs_rcv[i]-1
            p_snd = p+ptrs_snd[j]-1
            for k in 1:plength
              rcv[rcv_id].data[p_rcv,k] = snd_snd_id.data[p_snd,k]
            end
          end
        end
      end
      @async rcv
    end

    function PartitionedArrays.exchange_impl!(
      rcv::MPIArray,
      snd::MPIArray,
      graph::ExchangeGraph{<:MPIArray},
      ::Type{<:$T}
      )

      @assert size(rcv) == size(snd)
      @assert graph.rcv.comm === graph.rcv.comm
      @assert graph.rcv.comm === graph.snd.comm
      comm = graph.rcv.comm
      req_all = MPI.Request[]
      data_snd = JaggedArray(snd.item)
      data_rcv = rcv.item
      @assert isa(data_rcv,ParamJaggedArray)
      for (i,id_rcv) in enumerate(graph.rcv.item)
        rank_rcv = id_rcv-1
        ptrs_rcv = data_rcv.ptrs
        buff_rcv = view(data_rcv.data,ptrs_rcv[i]:(ptrs_rcv[i+1]-1),:)
        reqr = MPI.Irecv!(buff_rcv,rank_rcv,PartitionedArrays.EXCHANGE_IMPL_TAG,comm)
        push!(req_all,reqr)
      end
      for (i,id_snd) in enumerate(graph.snd.item)
        rank_snd = id_snd-1
        ptrs_snd = data_snd.ptrs
        buff_snd = view(data_snd.data,ptrs_snd[i]:(ptrs_snd[i+1]-1),:)
        reqs = MPI.Isend(buff_snd,rank_snd,PartitionedArrays.EXCHANGE_IMPL_TAG,comm)
        push!(req_all,reqs)
      end
      @async begin
        @static if isdefined(MPI,:Testall)
          while ! MPI.Testall(req_all)
            yield()
          end
        else
          while ! MPI.Testall!(req_all)[1]
            yield()
          end
        end
        rcv
      end
    end

    function PartitionedArrays.exchange_impl!(
      rcv::DebugArray,
      snd::DebugArray,
      graph::ExchangeGraph{<:DebugArray},
      ::Type{<:$T}
      )

      g = ExchangeGraph(graph.snd.items,graph.rcv.items)
      @async begin
        sleep(0.2)
        PartitionedArrays.exchange_impl!(rcv.items,snd.items,g,$T) |> wait
        rcv
      end
    end
  end
end

for T in (:AbstractArray,:ConsecutiveParamVector)
  @eval begin
    function PartitionedArrays.reduction!(op,b::MPIArray,a::MPIArray{<:$T{<:Number}};destination=PartitionedArrays.MAIN,init=nothing)
      @assert a.comm === b.comm
      comm = a.comm
      sdata = _data(a.item)
      rdata = similar(sdata)
      opr = MPI.Op(op,eltype(sdata))
      if destination !== :all
        root = destination-1
        MPI.Reduce!(sdata,rdata,opr,root,comm)
        if MPI.Comm_rank(comm) == root
          init !== nothing && (rdata .= op.(rdata,init))
          b.item = _rewrap(a.item,rdata)
        end
      else
        MPI.Allreduce!(sdata,rdata,opr,comm)
        init !== nothing && (rdata .= op.(rdata,init))
        b.item = _rewrap(a.item,rdata)
      end
      b
    end
  end
end

function PartitionedArrays.reduction!(
  op,
  b::MPIArray,
  a::MPIArray{<:ParamJaggedArray};
  destination=PartitionedArrays.MAIN,
  kwargs...
  )

  c = gather(a;destination)
  map!(i->reduce(op,i;kwargs...),b,c)
  b
end

sreduce(a) = reduce(+,a)
# like sreduce, but for values that are replicated (not partial) across the
# ranks contributing to a given entry: picks whichever rank's entry is nonzero,
# instead of summing
nzreduce(a) = reduce((x,y) -> ifelse.(iszero.(x),y,x),a)

# utils

_data(a::AbstractArray{<:Number}) = a
_data(a::ConsecutiveParamVector) = get_all_data(a)
_data(a::ParamJaggedArray) = a.data

_param_eachindex(a) = 1:_get_plength(a)
_get_plength(a) = size(_data(a),2)
_innersize(a) = size(_data(a),1)
_getindex(a,i,j) = getindex(_data(a),i,j)
_setindex!(a,v,i,j) = setindex!(_data(a),v,i,j)

function _getindex(a::OwnAndGhostParamVectors,i,j)
  n_own = innerlength(a.own_values)
  k = a.permutation[i]
  if k > n_own
    a.ghost_values.data[k-n_own,j]
  else
    a.own_values.data[k,j]
  end
end

function _setindex!(a::OwnAndGhostParamVectors,v,i,j)
  n_own = innerlength(a.own_values)
  k = a.permutation[i]
  if k > n_own
    a.ghost_values.data[k-n_own,j] = v
  else
    a.own_values.data[k,j] = v
  end
end

_rewrap(a::AbstractArray,b) = b
_rewrap(a::ConsecutiveParamArray,b) = ConsecutiveParamArray(b)