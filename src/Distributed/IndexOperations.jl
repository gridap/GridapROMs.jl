# general

flat_row_partition(a) = row_partition(a)

function flat_row_partition(a::PSparseMatrix)
  nnz_local = map(nnz,local_values(a))
  n_nz_global = reduce(+,nnz_local,init=0)
  variable_partition(nnz_local,n_nz_global)
end

row_partition(a) = a
row_partition(a::PVector) = a.index_partition
row_partition(a::PSparseMatrix) = a.row_partition
row_partition(a::GenericPArray) = a.index_partition
row_partition(a::PRange) = partition(a)

col_partition(a) = a
col_partition(a::PVector) = @notimplemented
col_partition(a::PSparseMatrix) = a.col_partition
col_partition(a::GenericPArray) = @notimplemented
col_partition(a::PRange) = partition(a)

# DOF maps

abstract type PartitionedDofMap <: AbstractVector{TrivialDofMap{Int}} end

struct PVectorDofMap{A<:PRange,B<:PRange} <: PartitionedDofMap
  rows::A
  rows_space::B
end

GridapDistributed.local_views(i::PVectorDofMap) = local_views(i.rows)
row_partition(i::PVectorDofMap) = row_partition(i.rows)

struct PSparseMatrixDofMap{A<:PRange,B<:PRange,C<:AbstractArray{<:SparsityPattern}} <: PartitionedDofMap
  rows::A
  cols::A
  rows_space::B
  cols_space::B
  loc_sparsity::C
end

function GridapDistributed.local_views(i::PSparseMatrixDofMap)
  map(local_views(i.loc_sparsity)) do sparsity
    TrivialSparseMatrixDofMap(sparsity)
  end
end

row_partition(i::PSparseMatrixDofMap) = row_partition(i.rows)
col_partition(i::PSparseMatrixDofMap) = col_partition(i.cols)

function flat_row_partition(a::PSparseMatrixDofMap)
  nnz_local = map(nnz,a.loc_sparsity)
  nnz_global = reduce(+,nnz_local,init=0)
  variable_partition(nnz_local,nnz_global)
end

struct BlockPVectorDofMap{A<:AbstractVector{<:PVectorDofMap},B<:Union{PRange,BlockPRange}} <: AbstractVector{PartitionedDofMap}
  blocks::A
  rows_space::B
end

Base.size(i::BlockPVectorDofMap) = size(i.blocks)
Base.getindex(i::BlockPVectorDofMap,j::Integer) = i.blocks[j]

struct BlockPSparseMatrixDofMap{A<:AbstractMatrix{<:PSparseMatrixDofMap},B<:Union{PRange,BlockPRange}} <: AbstractMatrix{PartitionedDofMap}
  blocks::A
  rows_space::B
  cols_space::B
end

Base.size(i::BlockPSparseMatrixDofMap) = size(i.blocks)
Base.getindex(i::BlockPSparseMatrixDofMap,j::Integer,k::Integer) = i.blocks[j,k]

function DofMaps.get_dof_map(f::DistributedSingleFieldFESpace)
  assem = SparseMatrixAssembler(f,f)
  rows,_,space_rows,_ = get_assembly_maps(assem)
  PVectorDofMap(rows,space_rows)
end

function DofMaps.get_dof_map(f::DistributedMultiFieldFESpace)
  blocks = map(get_dof_map,f.field_fe_space)
  BlockPVectorDofMap(blocks,get_free_dof_ids(f))
end

function DofMaps.get_sparse_dof_map(trial::DistributedSingleFieldFESpace,test::DistributedSingleFieldFESpace)
  assem = SparseMatrixAssembler(trial,test)
  rows,cols,space_rows,space_cols = get_assembly_maps(assem)
  loc_sparsity = map(get_sparsity,local_views(trial),local_views(test))
  PSparseMatrixDofMap(rows,cols,space_rows,space_cols,loc_sparsity)
end

function DofMaps.get_sparse_dof_map(trial::DistributedMultiFieldFESpace,test::DistributedMultiFieldFESpace)
  ntest = num_fields(test)
  ntrial = num_fields(trial)
  blocks = map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    get_sparse_dof_map(trial.field_fe_space[j],test.field_fe_space[i])
  end
  BlockPSparseMatrixDofMap(blocks,get_free_dof_ids(trial),get_free_dof_ids(test))
end

function DofMaps._get_dof_map(f::DistributedSingleFieldFESpace,b::PVector)
  rows, = axes(b)
  space_rows = get_free_dof_ids(f)
  PVectorDofMap(rows,space_rows)
end

function DofMaps._get_dof_map(f::DistributedMultiFieldFESpace,b::PVector)
  b′ = change_ghost(b,get_free_dof_ids(f))
  blocks = map(1:num_fields(f)) do i
    bi = restrict_to_field(f,b′,i)
    DofMaps._get_dof_map(f.field_fe_space[i],bi)
  end
  BlockPVectorDofMap(blocks,get_free_dof_ids(f))
end

function DofMaps._get_dof_map(f::DistributedMultiFieldFESpace{<:BlockMultiFieldStyle},b::PVector)
  DofMaps._get_dof_map(MultiFieldFESpace(f.field_fe_space),b)
end

function DofMaps._get_dof_map(f::DistributedMultiFieldFESpace,b::BlockPVector)
  field_blocks = map(DofMaps._get_dof_map,f.field_fe_space,blocks(b))
  BlockPVectorDofMap(field_blocks,get_free_dof_ids(f))
end

function DofMaps._get_sparse_dof_map(
  trial::DistributedSingleFieldFESpace,
  test::DistributedSingleFieldFESpace,
  A::PSparseMatrix
  )

  rows,cols = axes(A)
  space_rows = get_free_dof_ids(test)
  space_cols = get_free_dof_ids(trial)
  loc_sparsity = map(SparsityPattern,local_values(A))
  return PSparseMatrixDofMap(rows,cols,space_rows,space_cols,loc_sparsity)
end

function DofMaps._get_sparse_dof_map(
  trial::DistributedMultiFieldFESpace,
  test::DistributedMultiFieldFESpace,
  A::BlockPMatrix
  )

  ntest = num_fields(test)
  ntrial = num_fields(trial)
  blocks = map(Iterators.product(1:ntest,1:ntrial)) do (i,j)
    DofMaps._get_sparse_dof_map(trial[j],test[i],A[Block(i,j)])
  end
  BlockPSparseMatrixDofMap(blocks,get_free_dof_ids(test),get_free_dof_ids(trial))
end

function get_assembly_maps(assem::DistributedSparseMatrixAssembler)
  space_rows = get_rows(assem)
  space_cols = get_cols(assem)
  strategy = FESpaces.get_assembly_strategy(assem)
  builder = get_matrix_builder(assem)
  counter = nz_counter(builder,(space_rows,space_cols))
  alloc = nz_allocation(counter)
  rows,cols = get_assembly_maps(strategy,alloc)
  return (rows,cols,space_rows,space_cols)
end

function get_assembly_maps(::FullyAssembledRows,a::DistributedAllocationCOO)
  I,J, = get_allocations(a)
  row_gids = get_test_gids(a)
  col_gids = get_trial_gids(a)

  rows = _setup_prange(row_gids,I;ghost=false,ax=:rows)
  to_global_indices!(J,col_gids;ax=:cols)
  cols = _setup_prange(col_gids,J;ax=:cols)
  to_local_indices!(J,cols;ax=:cols)

  return (rows,cols)
end

function get_assembly_maps(::SubAssembledRows,a::DistributedAllocationCOO)
  I,J, = get_allocations(a)
  row_gids = get_test_gids(a)
  col_gids = get_trial_gids(a)

  to_global_indices!(I,row_gids;ax=:rows)
  to_global_indices!(J,col_gids;ax=:cols)

  Jo = get_gid_owners(J,col_gids;ax=:cols)
  rows = _setup_prange(row_gids,I;ax=:rows)
  cols = _setup_prange(col_gids,J;ax=:cols,owners=Jo)

  to_local_indices!(I,rows;ax=:cols)
  to_local_indices!(J,cols;ax=:cols)

  return (rows,cols)
end

function blockify(b::BlockPVector,i::BlockPVectorDofMap)
  b
end

function blockify(b::PVector,i::BlockPVectorDofMap)
  b′ = change_ghost(b,i.rows_space)
  nfields = length(i.blocks)
  rows = ntuple(k -> partition(i[k].rows_space),nfields)
  values = map(local_values(b′),rows...) do b′,ranks...
    lens = map(local_length,ranks)
    offsets = cumsum((0,lens...))
    ntuple(k -> get_param_entry(b′,offsets[k]+1:offsets[k+1]),nfields)
  end
  map(1:nfields) do k
    values_k = map(v -> v[k],values)
    indices_k = partition(i[k].rows_space)
    PVector(values_k,indices_k)
  end |> mortar
end

function blockify(A::BlockPMatrix,i::BlockPSparseMatrixDofMap)
  A
end

function blockify(A::PSparseMatrix,i::BlockPSparseMatrixDofMap)
  @notimplemented
end

# DEIM related

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

function RBSteady._evaluate!(a::VectorBlock,cellrows::VectorBlock,rows::LocalDofs)
  for i in eachindex(a)
    if a.touched[i]
      @check cellrows.touched[i]
      RBSteady._evaluate!(a.array[i],cellrows.array[i],rows)
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

function RBSteady._evaluate!(a::MatrixBlock,cellrows::VectorBlock,cellcols::VectorBlock,rows::LocalDofs,cols::LocalDofs)
  for j in axes(a,2), i in axes(a,1)
    if a.touched[i,j]
      @check cellrows.touched[i] && cellcols.touched[j]
      RBSteady._evaluate!(a.array[i,j],cellrows.array[i],cellcols.array[j],rows,cols)
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
  dof_maps::PSparseMatrixDofMap
  )

  row_parts = row_partition(dof_maps)
  col_parts = col_partition(dof_maps)
  r,c = map(sids,local_views(dof_maps),row_parts,col_parts) do i,dof_map,rowp,colp
    rci = i.index_parts
    li = _remap(i,global_to_local(rci))
    r,c = recast_split_indices(li,dof_map)
    _remap!(r,local_to_global(rowp))
    _remap!(c,local_to_global(colp))
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

  R′,C′ = map(row_parts,col_parts) do rowp,colp
    g2lr = global_to_local(rowp)
    g2lc = global_to_local(colp)
    ikeep,rkeep,ckeep = _keep_rows_and_cols(grows,gcols,g2lr,g2lc)
    R′ = LocalDofs(rkeep,copy(ikeep),rowp)
    C′ = LocalDofs(ckeep,copy(ikeep),colp)
    (R′,C′)
  end |> tuple_of_arrays

  return (R′,C′)
end

function DofMaps.sparsify_split_indices(
  frows::AbstractArray{<:LocalDofs},
  fcols::AbstractArray{<:LocalDofs},
  dof_maps::PSparseMatrixDofMap
  )

  nz_partition = flat_row_partition(dof_maps)
  map(local_views(frows),local_views(fcols),local_views(dof_maps),local_views(nz_partition)) do i,j,dof_map,nzids
    @check i.global_cols == j.global_cols
    rci = i.index_parts
    rcj = j.index_parts
    li = _remap(i,global_to_local(rci))
    lj = _remap(j,global_to_local(rcj))
    sl = sparsify_split_indices(li,lj,dof_map)
    _remap!(sl,local_to_global(nzids))
    # the result here is a list of global nz indices
    LocalDofs(sl,i.global_cols,nzids)
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
    x[i] = iszero(xi) ? xi : x_to_y[xi]
  end
end

function _remap(x,x_to_y)
  x′ = copy(x)
  _remap!(x′,x_to_y)
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