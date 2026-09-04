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

function flat_row_partition(a::PSparseMatrix)
  nnz_local = map(local_values(a)) do lval
    nnz(lval)
  end
  n_nz_global = reduce(+,nnz_local,init=0)
  nz_part = variable_partition(nnz_local,n_nz_global)
  map(nz_part,row_partition(a),col_partition(a)) do nzidx,lrow,lcol
    NZIndexPartition(nzidx,lrow,lcol)
  end
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