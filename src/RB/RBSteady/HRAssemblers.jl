function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_irows = get_cell_irows(interp)
  cell_icols = get_cell_icols(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_irows,cell_icols,icells)
end

function collect_reduced_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_irows = get_cell_irows(interp)
  cell_icols = get_cell_icols(interp)
  cells = get_integration_cells(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = lazy_map(Reindex(get_contribution(a,strian)),cells)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2 "$(ndims(eltype(cell_mat)))"
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_irows,cell_icols,icells)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_irows = get_cell_irows(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_irows,icells)
end

function collect_reduced_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_irows = get_cell_irows(interp)
  cells = get_integration_cells(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = lazy_map(Reindex(get_contribution(a,strian)),cells)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_irows,icells)
end

struct AddHREntriesMap{F} <: Map
  combine::F
end

function Arrays.return_cache(k::AddHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),param_length(vs))
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v,i,j)
  add_hr_entries!(cache,k.combine,A,v,i,j)
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v,i)
  add_hr_entries!(cache,k.combine,A,v,i)
end

function Arrays.return_cache(k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
  qs = findall(v.touched)
  i,j = Tuple(first(qs))
  cij = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
  ni,nj = size(v.touched)
  cache = Matrix{typeof(cij)}(undef,ni,nj)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        cache[i,j] = return_cache(k,A,v.array[i,j],I.array[i],J.array[j])
      end
    end
  end
  cache
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
  ni,nj = size(v.touched)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        evaluate!(cache[i,j],k,A,v.array[i,j],I.array[i],J.array[j])
      end
    end
  end
end

function Arrays.return_cache(k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock)
  qs = findall(v.touched)
  i = first(qs)
  ci = return_cache(k,A,v.array[i],I.array[i])
  ni = length(v.touched)
  cache = Vector{typeof(ci)}(undef,ni)
  for i in 1:ni
    if v.touched[i]
      cache[i] = return_cache(k,A,v.array[i],I.array[i])
    end
  end
  cache
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v::VectorBlock,I::VectorBlock)
  ni = length(v.touched)
  for i in 1:ni
    if v.touched[i]
      evaluate!(cache[i],k,A,v.array[i],I.array[i])
    end
  end
end

for MT in (:MatrixBlock,:MatrixBlockView)
  Aij = (MT == :MatrixBlock) ? :(A.array[i,j]) : :(A[i,j])
  @eval begin
    function Arrays.return_cache(k::AddHREntriesMap,A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
      qs = findall(v.touched)
      i,j = Tuple(first(qs))
      cij = return_cache(k,$Aij,v.array[i,j],I.array[i],J.array[j])
      ni,nj = size(v.touched)
      cache = Matrix{typeof(cij)}(undef,ni,nj)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            cache[i,j] = return_cache(k,$Aij,v.array[i,j],I.array[i],J.array[j])
          end
        end
      end
      cache
    end

    function Arrays.evaluate!(cache,k::AddHREntriesMap,A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,$Aij,v.array[i,j],I.array[i],J.array[j])
          end
        end
      end
    end
  end 
end 

for VT in (:VectorBlock,:VectorBlockView)
  Ai = (VT == :VectorBlock) ? :(A.array[i]) : :(A[i])
  @eval begin
    function Arrays.return_cache(k::AddHREntriesMap,A::$VT,v::VectorBlock,I::VectorBlock)
      qs = findall(v.touched)
      i = first(qs)
      ci = return_cache(k,$Ai,v.array[i],I.array[i])
      ni = length(v.touched)
      cache = Vector{typeof(ci)}(undef,ni)
      for i in 1:ni
        if v.touched[i]
          cache[i] = return_cache(k,$Ai,v.array[i],I.array[i])
        end
      end
      cache
    end

    function Arrays.evaluate!(cache,k::AddHREntriesMap,A::$VT,v::VectorBlock,I::VectorBlock)
      ni = length(v.touched)
      for i in 1:ni
        if v.touched[i]
          evaluate!(cache[i],k,$Ai,v.array[i],I.array[i])
        end
      end
    end
  end 
end

@inline function add_hr_entries!(vi,combine,b,vs,is)
  Algebra._add_entries!(combine,b,vs,is)
end

@inline function add_hr_entries!(vi,combine,b,vs::ParamBlock,is)
  Algebra._add_entries!(vi,combine,b,vs,is)
end

@inline function add_hr_entries!(vij,combine,A,vs,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          if i == j
            vij = vs[li,lj]
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

@inline function add_hr_entries!(vij,combine,A,vs::ParamBlock,is,js)
  for (lj,j) in enumerate(js)
    if j>0
      for (li,i) in enumerate(is)
        if i>0
          if i == j
            get_param_entry!(vij,vs,li,lj)
            add_entry!(combine,A,vij,i)
          end
        end
      end
    end
  end
  A
end

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

function assemble_hr_vector_add!(
  b::ArrayBlock,
  cellvec,
  cellidsrows::ArrayBlock,
  icells::ArrayBlock
  )
  @check cellidsrows.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(FetchBlockMap(cellvec,i),icells.array[i])
      assemble_hr_vector_add!(b.array[i],cellveci,cellidsrows.array[i],icells.array[i])
    end
  end
end

function assemble_hr_vector_add!(b,cellvec,cellidsrows,icells)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    _numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
end

@noinline function _numeric_loop_hr_vector!(vec,caches,cell_vals,cell_rows)
  add!,add_cache,vals_cache,rows_cache = caches
  @assert length(cell_vals) == length(cell_rows)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,vec,vals,rows)
  end
end

function assemble_hr_matrix_add!(
  A::ArrayBlock,
  cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  icells::ArrayBlock
  )
  @check cellidsrows.touched == cellidscols.touched == icells.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(FetchBlockMap(cellmat,i),icells.array[i])
      assemble_hr_matrix_add!(
        A.array[i],cellmati,cellidsrows.array[i],cellidscols.array[i],icells.array[i])
    end
  end
end

function assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,icells)
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
    _numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end

@noinline function _numeric_loop_hr_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add!,add_cache,vals_cache,rows_cache,cols_cache = caches
  @assert length(cell_vals) == length(cell_rows) == length(cell_cols)
  for cell in 1:length(cell_rows)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end
