function PartitionedArrays.allocate_gather_impl(snd,destination,::Type{T}) where T<:AbstractParamVector
  l = map(innerlength,snd)
  l_dest = gather(l;destination)
  S = eltype2(T)
  function f(l,snd)
    ptrs = length_to_ptrs!(pushfirst!(l,one(eltype(l))))
    ndata = ptrs[end]-1
    data = Vector{S}(undef,ndata)
    plength = param_length(snd)
    pdata = global_parameterize(data,plength)
    ParamJaggedArray{S,Int32}(pdata,ptrs)
  end
  if isa(destination,Integer)
    function g(l,snd)
      ptrs = Vector{Int32}(undef,1)
      data = Vector{S}(undef,0)
      plength = param_length(snd)
      pdata = global_parameterize(data,plength)
      ParamJaggedArray(pdata,ptrs)
    end
    rcv = map_main(f,l_dest,snd;otherwise=g,main=destination)
  else
    @assert destination === :all
    rcv = map(f,l_dest,snd)
  end
  rcv
end

function PartitionedArrays.allocate_scatter_impl(snd,source,::Type{T}) where T<:AbstractParamVector
  S = eltype2(T)
  counts,plength = map(snd) do snd
    innerlength(snd),param_length(snd)
  end |> tuple_of_arrays
  counts_scat = scatter(counts;source)
  map(counts_scat,plength) do count,plength
    data = Vector{S}(undef,count)
    global_parameterize(data,plength)
  end
end

function PartitionedArrays.assemble_impl!(
  f,
  vector_partition,
  cache,
  ::Type{<:ParamVectorAssemblyCache})

  buffer_snd = map(vector_partition,cache) do values,cache
    δ = _get_delta(cache.buffer_snd)
    local_indices_snd = cache.local_indices_snd
    for (p,lid) in enumerate(local_indices_snd.data)
      for i in param_eachindex(values)
        cache.buffer_snd.data[p+(i-1)*δ] = _getindex(values,lid,i)
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
      δ = _get_delta(cache.buffer_rcv)
      local_indices_rcv = cache.local_indices_rcv
      for (p,lid) in enumerate(local_indices_rcv.data)
        for i in param_eachindex(values)
          v = f(_getindex(values,lid,i),cache.buffer_rcv.data[p+(i-1)*δ])
          _setindex!(values,v,lid,i)
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

function PartitionedArrays.allocate_exchange_impl(snd,graph,::Type{T}) where T<:AbstractParamVector
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
    pdata = global_parameterize(data,plength)
    ParamJaggedArray(pdata,ptrs)
  end
  rcv
end

#

function PartitionedArrays.allocate_gather_impl(snd,destination,::Type{T}) where T<:AbstractMatrix
  l = map(innerlength,snd)
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

function PartitionedArrays.allocate_scatter_impl(snd,source,::Type{T}) where T<:AbstractMatrix
  S = eltype(T)
  counts,plength = map(snd) do snd
    size(snd,1),_get_plength(snd)
  end |> tuple_of_arrays
  counts_scat = scatter(counts;source)
  map(counts_scat,plength) do count,plength
    Matrix{S}(undef,count,plength)
  end
end

function PartitionedArrays.allocate_exchange_impl(snd,graph,::Type{T}) where T<:AbstractMatrix
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

for T in (:AbstractMatrix,:ConsecutiveParamVector)
  @eval begin
    function PartitionedArrays.exchange_impl!(rcv,snd,graph,::Type{<:$T})
      @assert PartitionedArrays.is_consistent(graph)
      @assert eltype(rcv) <: ParamJaggedArray
      snd_ids = graph.snd
      rcv_ids = graph.rcv
      @assert length(rcv_ids) == length(rcv)
      @assert length(rcv_ids) == length(snd)
      for rcv_id in 1:length(rcv_ids)
        for (i,snd_id) in enumerate(rcv_ids[rcv_id])
          snd_snd_id = JaggedArray(snd[snd_id])
          j = first(findall(k->k==rcv_id,snd_ids[snd_id]))
          ptrs_rcv = rcv[rcv_id].ptrs
          ptrs_snd = snd_snd_id.ptrs
          plength = _get_plength(snd[snd_id])
          δ_snd = length(ptrs_snd)-1
          δ_rcv = length(ptrs_rcv)-1
          @assert ptrs_rcv[i+1]-ptrs_rcv[i] == ptrs_snd[j+1]-ptrs_snd[j]
          for p in 1:(ptrs_rcv[i+1]-ptrs_rcv[i])
            p_rcv = p+ptrs_rcv[i]-1
            p_snd = p+ptrs_snd[j]-1
            for i in 1:plength
              rcv[rcv_id].data[p_rcv+(i-1)*δ_rcv] = snd_snd_id.data[p_snd+(i-1)*δ_snd]
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
        yield()
        PartitionedArrays.exchange_impl!(rcv.items,snd.items,g,$T) |> wait
        rcv
      end
    end
  end
end

# utils

_get_plength(a::AbstractParamArray) = param_length(a)
_get_plength(a::ParamJaggedArray) = param_length(a)
_get_plength(a::AbstractMatrix) = size(a,2)

_getindex(a::ConsecutiveParamVector,i,j) = a.data[i,j]
_setindex!(a::ConsecutiveParamVector,v,i,j) = (a.data[i,j] = v)

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
