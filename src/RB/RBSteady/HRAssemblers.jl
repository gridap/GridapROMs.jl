function collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_idofs,icells)
end

function collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_idofs,icells)
end

struct AddHREntriesMap{F} <: Map
  combine::F
end

function Arrays.return_cache(k::AddHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),param_length(vs))
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v,i)
  add_hr_entries!(cache,k.combine,A,v,i)
end

function Arrays.return_cache(k::AddHREntriesMap,A,v::MatrixBlock,IJ::MatrixBlock)
  qs = findall(v.touched)
  i,j = Tuple(first(qs))
  cij = return_cache(k,A,v.array[i,j],IJ.array[i,j])
  ni,nj = size(v.touched)
  cache = Matrix{typeof(cij)}(undef,ni,nj)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        cache[i,j] = return_cache(k,A,v.array[i,j],IJ.array[i,j])
      end
    end
  end
  cache
end

function Arrays.evaluate!(cache,k::AddHREntriesMap,A,v::MatrixBlock,IJ::MatrixBlock)
  ni,nj = size(v.touched)
  for j in 1:nj
    for i in 1:ni
      if v.touched[i,j]
        evaluate!(cache[i,j],k,A,v.array[i,j],IJ.array[i,j])
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
    function Arrays.return_cache(k::AddHREntriesMap,A::$MT,v::MatrixBlock,IJ::MatrixBlock)
      qs = findall(v.touched)
      i,j = Tuple(first(qs))
      cij = return_cache(k,$Aij,v.array[i,j],IJ.array[i,j])
      ni,nj = size(v.touched)
      cache = Matrix{typeof(cij)}(undef,ni,nj)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            cache[i,j] = return_cache(k,$Aij,v.array[i,j],IJ.array[i,j])
          end
        end
      end
      cache
    end

    function Arrays.evaluate!(cache,k::AddHREntriesMap,A::$MT,v::MatrixBlock,IJ::MatrixBlock)
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,$Aij,v.array[i,j],IJ.array[i,j])
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

@inline function add_hr_entries!(vi,combine,A,vs,is)
  Algebra._add_entries!(combine,A,vs,is)
end

@inline function add_hr_entries!(vi,combine,A,vs::ParamBlock,is)
  Algebra._add_entries!(vi,combine,A,vs,is)
end

assemble_hr_array_add!(A,celldata::Tuple) = assemble_hr_array_add!(A,celldata...)

function assemble_hr_array_add!(
  A::ArrayBlock,
  _cellvals,
  celldofs::ArrayBlock,
  icells::ArrayBlock
  )

  @check celldofs.touched == icells.touched
  for i in eachindex(celldofs)
    if celldofs.touched[i]
      isempty(icells[i]) && continue
      cellvalsi = lazy_map(FetchBlockMap(_cellvals,i),icells[i])
      _assemble_hr_array_add!(A[i],cellvalsi,celldofs[i])
    end
  end
  A
end

function assemble_hr_array_add!(A,_cellvals,celldofs,icells)
  isempty(icells) && return A
  cellvals = lazy_map(Reindex(_cellvals),icells)
  _assemble_hr_array_add!(A,cellvals,celldofs)
  A
end

function _assemble_hr_array_add!(A,cellvals,celldofs)
  if length(cellvals) > 0
    dofs_cache = array_cache(celldofs)
    vals_cache = array_cache(cellvals)
    vals1 = getindex!(vals_cache,cellvals,1)
    rows1 = getindex!(dofs_cache,celldofs,1)
    add! = AddHREntriesMap(+)
    add_cache = return_cache(add!,A,vals1,rows1)
    caches = add!,add_cache,vals_cache,dofs_cache
    _numeric_loop_hr_array!(A,caches,cellvals,celldofs)
  end
  A
end

@noinline function _numeric_loop_hr_array!(arr,caches,cell_vals,cell_dofs)
  add!,add_cache,vals_cache,dofs_cache = caches
  @assert length(cell_vals) == length(cell_dofs)
  for cell in 1:length(cell_dofs)
    dofs = getindex!(dofs_cache,cell_dofs,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,arr,vals,dofs)
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