function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )
  
  cell_row_ids = get_cell_row_ids(interp)
  cell_col_ids = get_cell_col_ids(interp)
  rows = get_interpolation_rows(interp)
  cols = get_interpolation_cols(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_row_ids,cell_col_ids,rows,cols,icells)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_row_ids = get_cell_row_ids(interp)
  rows = get_interpolation_rows(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_row_ids,rows,icells)
end

struct AddHREntriesMap{F} <: Map
  combine::F
end

function Arrays.return_cache(k::AddHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),param_length(vs))
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v,i,j,r,c)
  add_hr_entries!(cache,k.combine,A,v,i,j,r,c)
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v,i,r)
  add_hr_entries!(cache,k.combine,A,v,i,r)
end

function Arrays.return_cache(k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock,r,c)
  qs = findall(v.touched)
  i,j = Tuple(first(qs))
  cij = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
  ni,nj = size(v.touched)
  cache = Matrix{typeof(cij)}(undef,ni,nj)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        cache[i,j] = return_cache(k,A,v.array[i,j],I.array[i],J.array[j],r,c)
      end
    end
  end
  cache
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock,r,c)
  ni,nj = size(v.touched)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        evaluate!(cache[i,j],k,A,v.array[i,j],I.array[i],J.array[j],r,c)
      end
    end
  end
end

function Arrays.return_cache(k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock,r)
  qs = findall(v.touched)
  i = first(qs)
  ci = return_cache(k,A,v.array[i],I.array[i])
  ni = length(v.touched)
  cache = Vector{typeof(ci)}(undef,ni)
  for i in 1:ni
    if v.touched[i]
      cache[i] = return_cache(k,A,v.array[i],I.array[i],r)
    end
  end
  cache
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock,r)
  ni = length(v.touched)
  for i in 1:ni
    if v.touched[i]
      evaluate!(cache[i],k,A,v.array[i],I.array[i],r)
    end
  end
end

for MT in (:MatrixBlock,:MatrixBlockView)
  Aij = (MT == :MatrixBlock) ? :(A.array[i,j]) : :(A[i,j])
  @eval begin
    function Arrays.return_cache(k::AddHREntriesMap,A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock,r,c)
      qs = findall(v.touched)
      i,j = Tuple(first(qs))
      cij = return_cache(k,$Aij,v.array[i,j],I.array[i],J.array[j])
      ni,nj = size(v.touched)
      cache = Matrix{typeof(cij)}(undef,ni,nj)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            cache[i,j] = return_cache(k,$Aij,v.array[i,j],I.array[i],J.array[j],r,c)
          end
        end
      end
      cache
    end

    function Arrays.evaluate!(cache,k::AddHREntriesMap,A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock,r,c)
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,$Aij,v.array[i,j],I.array[i],J.array[j],r,c)
          end
        end
      end
    end
  end 
end 

for VT in (:VectorBlock,:VectorBlockView)
  Ai = (VT == :VectorBlock) ? :(A.array[i]) : :(A[i])
  @eval begin
    function Arrays.return_cache(k::AddHREntriesMap,A::$VT,v::VectorBlock,I::VectorBlock,r)
      qs = findall(v.touched)
      i = first(qs)
      ci = return_cache(k,$Ai,v.array[i],I.array[i],r)
      ni = length(v.touched)
      cache = Vector{typeof(ci)}(undef,ni)
      for i in 1:ni
        if v.touched[i]
          cache[i] = return_cache(k,$Ai,v.array[i],I.array[i],r)
        end
      end
      cache
    end

    function Arrays.evaluate!(cache,k::AddHREntriesMap,A::$VT,v::VectorBlock,I::VectorBlock,r)
      ni = length(v.touched)
      for i in 1:ni
        if v.touched[i]
          evaluate!(cache[i],k,$Ai,v.array[i],I.array[i],r)
        end
      end
    end
  end 
end

@inline function add_hr_entries!(vi,combine,b,vs,is,r)
  for (li,i) in enumerate(is)
    ir = _indexin(i,r)
    if !isnothing(ir)
      vi = vs[li]
      add_entry!(combine,b,vi,i)
    end
  end  
  b
end

@inline function add_hr_entries!(vi,combine,b,vs::ParamBlock,is,r)
  for (li,i) in enumerate(is)
    ir = _indexin(i,r)
    if !isnothing(ir)
      get_param_entry!(vi,vs,li)
      add_entry!(combine,b,vi,i)
    end
  end  
  b
end

@inline function add_hr_entries!(vij,combine,A,vs,is,js,r,c)
  for (lj,j) in enumerate(js)
    ic = _indexin(j,c)
    if !isnothing(ic)
      for (li,i) in enumerate(is)
        ir = _indexin(i,r)
        if !isnothing(ir)
          if ir == ic
            vij = vs[li,lj]
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

@inline function add_hr_entries!(vij,combine,A,vs::ParamBlock,is,js,r,c)
  for (lj,j) in enumerate(js)
    ic = _indexin(j,c)
    if !isnothing(ic)
      for (li,i) in enumerate(is)
        ir = _indexin(i,r)
        if !isnothing(ir)
          if ir == ic
            get_param_entry!(vij,vs,li,lj)
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

function assemble_hr_vector_add!(
  b::ArrayBlock,
  _cellvec,
  cellidsrows::ArrayBlock,
  rows::ArrayBlock,
  icells::ArrayBlock
  )

  @check cellidsrows.touched == rows.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(FetchBlockMap(_cellvec,i),icells[i])
      _assemble_hr_vector_add!(b[i],cellveci,cellidsrows[i],rows[i])
    end
  end
  b
end

function assemble_hr_vector_add!(b,_cellvec,cellidsrows,rows,icells)
  cellvec = lazy_map(Reindex(_cellvec),icells)
  _assemble_hr_vector_add!(b,cellvec,cellidsrows,rows)
  b
end

function _assemble_hr_vector_add!(b,cellvec,cellidsrows,rows)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    _numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows,rows)
  end
  b
end

@noinline function _numeric_loop_hr_vector!(vec,caches,cell_vals,cell_rows,hr_rows)
  add!,add_cache,vals_cache,rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows,hr_rows)
  end
end

function assemble_hr_matrix_add!(
  A::ArrayBlock,
  _cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  rows::ArrayBlock,
  cols::ArrayBlock,
  icells::ArrayBlock
  )
  
  @check cellidsrows.touched == cellidscols.touched == rows.touched == cols.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(FetchBlockMap(_cellmat,i),icells[i])
      _assemble_hr_matrix_add!(A[i],cellmati,cellidsrows[i],cellidscols[i],rows[i],cols[i])
    end
  end
  A
end

function assemble_hr_matrix_add!(A,_cellmat,cellidsrows,cellidscols,rows,cols,icells)
  cellmat = lazy_map(Reindex(_cellmat),icells)
  _assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,rows,cols)
  A
end

function _assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,rows,cols)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    vals1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,A,vals1,rows1,cols1)
    caches = add!,add_cache,vals_cache,rows_cache,cols_cache
    _numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols,rows,cols)
  end
  A
end

@noinline function _numeric_loop_hr_matrix!(mat,caches,cell_vals,cell_rows,cell_cols,hr_rows,hr_cols)
  add!,add_cache,vals_cache,rows_cache,cols_cache = caches
  @assert length(cell_vals) == length(cell_rows) == length(cell_cols)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols,hr_rows,hr_cols)
  end
end

# utils 

struct FetchBlockMap{A} <: Map
  values::A
  blockid::Int
end

function Arrays.return_cache(k::FetchBlockMap,i...)
  array_cache(k.values)
end

function Arrays.evaluate!(cache,k::FetchBlockMap,i...)
  a = getindex!(cache,k.values,i...)
  a.array[k.blockid]
end

function _indexin(j::Int,ids::AbstractVector{<:Integer})
  i = findfirst(==(j),ids)
  return i
end