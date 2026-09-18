struct NZIndexPartition{I<:AbstractLocalIndices,R<:AbstractLocalIndices,C<:AbstractLocalIndices} <: AbstractLocalIndices
  nz::I
  row::R
  col::C
end

PartitionedArrays.part_id(a::NZIndexPartition) = part_id(a.nz)
PartitionedArrays.local_to_global(a::NZIndexPartition) = local_to_global(a.nz)
PartitionedArrays.local_to_owner(a::NZIndexPartition) = local_to_owner(a.nz)
PartitionedArrays.own_to_global(a::NZIndexPartition) = own_to_global(a.nz)
PartitionedArrays.ghost_to_global(a::NZIndexPartition) = ghost_to_global(a.nz)
PartitionedArrays.ghost_to_owner(a::NZIndexPartition) = ghost_to_owner(a.nz)
PartitionedArrays.own_to_local(a::NZIndexPartition) = own_to_local(a.nz)
PartitionedArrays.ghost_to_local(a::NZIndexPartition) = ghost_to_local(a.nz)
PartitionedArrays.global_to_own(a::NZIndexPartition) = global_to_own(a.nz)
PartitionedArrays.global_to_local(a::NZIndexPartition) = global_to_local(a.nz)
PartitionedArrays.global_to_ghost(a::NZIndexPartition) = global_to_ghost(a.nz)
PartitionedArrays.own_length(a::NZIndexPartition) = own_length(a.nz)
PartitionedArrays.assembly_cache(a::NZIndexPartition) = PartitionedArrays.assembly_cache(a.nz)

function nz_partition(nz_part,row_partition,col_partition)
  map(nz_part,row_partition,col_partition) do nzidx,lrow,lcol
    NZIndexPartition(nzidx,lrow,lcol)
  end
end

function flat_row_partition(a::PSparseMatrix)
  nnz_local = map(nnz,local_values(a))
  n_nz_global = reduce(+,nnz_local,init=0)
  nz_part = variable_partition(nnz_local,n_nz_global)
  nz_partition(nz_part,row_partition(a),col_partition(a))
end

flat_row_partition(a) = row_partition(a)
flat_row_partition(a::AbstractArray{<:NZIndexPartition}) = a

row_partition(a) = a
row_partition(a::PVector) = a.index_partition
row_partition(a::PSparseMatrix) = a.row_partition
row_partition(a::NZIndexPartition) = a.row
row_partition(a::AbstractArray{<:NZIndexPartition}) = map(row_partition,a)

col_partition(a) = a
col_partition(a::PVector) = @notimplemented
col_partition(a::PSparseMatrix) = a.col_partition
col_partition(a::NZIndexPartition) = a.col
col_partition(a::AbstractArray{<:NZIndexPartition}) = map(col_partition,a)

struct LocalDofs{Tr,Tc,A<:AbstractLocalIndices} <: AbstractVector{Tr}
  global_rows::Vector{Tr}
  global_cols::Vector{Tc}
  index_parts::A
end

LocalDofs(index_parts) = LocalDofs(Int[],Int[],index_parts)

Base.size(a::LocalDofs) = size(a.global_rows)
Base.IndexStyle(::Type{<:LocalDofs}) = IndexLinear()
Base.getindex(a::LocalDofs,i::Int) = getindex(a.global_rows,i)
Base.setindex!(a::LocalDofs,v,i::Int) = setindex!(a.global_rows,v,i)
Base.copy(a::LocalDofs) = LocalDofs(copy(a.global_rows),copy(a.global_cols),a.index_parts)

function RBSteady._evaluate!(a,cellrows,rows::LocalDofs)
  fill!(a,zero(eltype(a)))
  for (irow,row) in enumerate(rows)
    for (icellrow,cellrow) in enumerate(cellrows)
      if row == cellrow
        a[icellrow] = rows.global_cols[irow]
      end
    end
  end
  a
end

function RBSteady._evaluate!(a,cellrows,cellcols,rows::LocalDofs,cols::LocalDofs)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for (irowcol,rowcol) in enumerate(zip(rows,cols))
    row,col = rowcol
    for (icellrow,cellrow) in enumerate(cellrows)
      for (icellcol,cellcol) in enumerate(cellcols)
        if row == cellrow && col == cellcol
          icellrowcol = icellrow + (icellcol-1)*ncellrows
          a[icellrowcol] = rows.global_cols[irowcol]
        end
      end
    end
  end
  a 
end

function DofMaps.recast_split_indices(
  sids::AbstractArray{<:LocalDofs},
  dof_maps::AbstractArray{<:AbstractDofMap}
  ) 

  sids
end

function DofMaps.recast_split_indices(
  sids::AbstractArray{<:LocalDofs},
  dof_maps::AbstractArray{<:AbstractSparseDofMap}
  )

  r,c = map(sids,dof_maps) do i,dof_map
    rci = i.index_parts
    li = _remap(i,global_to_local(rci))
    r,c = recast_split_indices(li,dof_map)
    _remap!(r,local_to_global(row_partition(rci)))
    _remap!(c,local_to_global(col_partition(rci)))
    (r,c)
  end |> tuple_of_arrays

  local_max = map(i -> isempty(i.global_cols) ? 0 : maximum(i.global_cols),sids)
  n = reduce(max,local_max)
  rcache,ccache = map(r,c) do r,c
    rcache = zeros(Int,n)
    ccache = zeros(Int,n)
    for (k,sk) in enumerate(r.global_cols)
      rcache[sk] = r[k]
      ccache[sk] = c[k]
    end
    (rcache,ccache)
  end |> tuple_of_arrays

  op(a,b) = max.(a,b) # assign a DOF to one and only one rank
  grows = reduce(op,rcache)
  gcols = reduce(op,ccache)

  R′,C′ = map(sids) do i
    rci = i.index_parts
    g2lr = global_to_local(row_partition(rci))
    g2lc = global_to_local(col_partition(rci))
    ikeep,rkeep,ckeep = _keep_rows_and_cols(grows,gcols,g2lr,g2lc)
    R′ = LocalDofs(rkeep,copy(ikeep),row_partition(rci))
    C′ = LocalDofs(ckeep,copy(ikeep),col_partition(rci))
    (R′,C′)
  end |> tuple_of_arrays

  return (R′,C′)
end

function DofMaps.sparsify_split_indices(
  frows::AbstractArray{<:LocalDofs},
  fcols::AbstractArray{<:LocalDofs},
  dof_maps::AbstractArray{<:AbstractSparseDofMap}
  )

  nnz_local = map(d -> nnz(get_sparsity(d)),dof_maps)
  n_nz_global = reduce(+,nnz_local,init=0)
  nz_part = variable_partition(nnz_local,n_nz_global)
  map(frows,fcols,dof_maps,nz_part) do i,j,dof_map,nzidx
    @check i.global_cols == j.global_cols
    rci = i.index_parts
    rcj = j.index_parts
    li = _remap(i,global_to_local(rci))
    lj = _remap(j,global_to_local(rcj))
    sl = sparsify_split_indices(li,lj,dof_map)
    _remap!(sl,local_to_global(nzidx))
    nzparts = NZIndexPartition(nzidx,rci,rcj)
    # the result here is a list of global nz indices
    LocalDofs(sl,i.global_cols,nzparts)
  end
end

for T in (:AbstractSparseMatrix,:SubSparseMatrix)
  @eval begin
    function DofMaps.recast_split_indices(sids::LocalDofs,a::$T)
      rids,cids = recast_split_indices(sids.global_rows,a)
      r = LocalDofs(rids,copy(sids.global_cols),sids.index_parts)
      c = LocalDofs(cids,copy(sids.global_cols),sids.index_parts)
      (r,c)
    end

    function DofMaps.sparsify_split_indices(frows::LocalDofs,fcols::LocalDofs,a::$T)
      sparsify_split_indices(frows.global_rows,fcols.global_rows,a)
    end
  end
end

function DofMaps.recast_split_indices(sids::AbstractArray,a::SubSparseMatrix)
  frows = similar(sids)
  fcols = similar(sids)
  fill!(frows,zero(eltype(frows)))
  fill!(fcols,zero(eltype(fcols)))
  prows,pcols = a.indices
  I,J, = findnz(a.parent)
  for (i,nzi) in enumerate(sids)
    if nzi > 0
      frows[i] = prows[I[nzi]]
      fcols[i] = pcols[J[nzi]]
    end
  end
  return frows,fcols
end

function DofMaps.sparsify_split_indices(frows::AbstractArray,fcols::AbstractArray,a::SubSparseMatrix)
  @assert length(frows) == length(fcols)
  sids = similar(frows)
  fill!(sids,zero(eltype(sids)))
  irows,icols = a.inv_indices
  for j in eachindex(frows)
    jrow = irows[frows[j]]
    jcol = icols[fcols[j]]
    sids[j] = nz_index(a.parent,jrow,jcol)
  end
  return sids
end

# utils 

function _remap!(x,x_to_y)
  for (i,xi) in enumerate(x)
    x[i] = x_to_y[xi]
  end
end

function _remap(x,x_to_y)
  x′ = copy(x)
  _remap!(x′, x_to_y)
  x′
end

function _keep_rows_and_cols(rows,cols,rowmap,colmap)
  @check length(rows) == length(cols)
  count = 0
  for (r,c) in zip(rows,cols)
    if !iszero(rowmap[r]) && !iszero(colmap[c])
      count += 1
    end
  end
  ikeep = zeros(Int,count)
  rkeep = zeros(Int,count)
  ckeep = zeros(Int,count)
  count = 0
  for (i,(r,c)) in enumerate(zip(rows,cols))
    if !iszero(rowmap[r]) && !iszero(colmap[c])
      count += 1
      ikeep[count] = i
      rkeep[count] = r
      ckeep[count] = c
    end
  end
  return ikeep,rkeep,ckeep
end