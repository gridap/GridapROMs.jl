function RBSteady.collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation,
  common_indices::AbstractVector
  )

  cell_row_ids = get_cell_row_ids(interp)
  cell_col_ids = get_cell_col_ids(interp)
  rows = get_interpolation_rows(interp)
  cols = get_interpolation_cols(interp)
  icells = get_owned_icells(interp,strian)
  locations = get_param_itimes(interp,common_indices)
  style = get_domain_style(interp)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_row_ids,cell_col_ids,rows,cols,icells,locations,style)
end

function RBSteady.collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation,
  common_indices::AbstractVector
  )

  cell_row_ids = get_cell_row_ids(interp)
  rows = get_interpolation_rows(interp)
  icells = get_owned_icells(interp,strian)
  locations = get_param_itimes(interp,common_indices)
  style = get_domain_style(interp)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_row_ids,rows,icells,locations,style)
end

function get_hr_param_entry!(v::AbstractVector,b::GenericParamBlock,hr_indices,i...)
  for (k,hrk) in enumerate(hr_indices)
    v[k] = b.data[hrk][i...]
  end
  v
end

function get_hr_param_entry!(v::AbstractVector,b::TrivialParamBlock,hr_indices,i...)
  vk = b.data[i...]
  fill!(v,vk)
end

struct AddTransientHREntriesMap{A<:TransientIntegrationDomainStyle,F,Is,It} <: Map
  style::A
  combine::F
  indices::Is
  locations::It
end

function AddTransientHREntriesMap(style::TransientIntegrationDomainStyle,indices,locations)
  AddTransientHREntriesMap(style,+,indices,locations)
end

get_param_time_inds(k::AddTransientHREntriesMap) = k.locations
get_param_inds(k::AddTransientHREntriesMap) = k.locations.axis1
get_time_inds(k::AddTransientHREntriesMap) = k.locations.axis2

function Arrays.return_cache(k::AddTransientHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(get_param_time_inds(k)))
end

for (T,f) in zip((:KroneckerDomain,:SequentialDomain),(:add_hr_kron_entries!,:add_hr_lin_entries!))
  @eval begin
    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},b,vs,is)
      $f(cache,k.combine,b,vs,is,k.indices,k.locations)
    end

    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,vs,is,js)
      r,c = k.indices
      $f(cache,k.combine,A,vs,is,js,r,c,k.locations)
    end
  end
end

for T in (:KroneckerDomain,:SequentialDomain)
  @eval begin
    function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
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

    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,A,v.array[i,j],I.array[i],J.array[j])
          end
        end
      end
    end

    function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A,v::VectorBlock,I::VectorBlock)
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

    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,v::VectorBlock,I::VectorBlock)
      ni = length(v.touched)
      for i in 1:ni
        if v.touched[i]
          evaluate!(cache[i],k,A,v.array[i],I.array[i])
        end
      end
    end
  end

  for MT in (:MatrixBlock,:MatrixBlockView)
    Aij = (MT == :MatrixBlock) ? :(A.array[i,j]) : :(A[i,j])
    @eval begin
      function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
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

      function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A::$MT,v::MatrixBlock,I::VectorBlock,J::VectorBlock)
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
      function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A::$VT,v::VectorBlock,I::VectorBlock)
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

      function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A::$VT,v::VectorBlock,I::VectorBlock)
        ni = length(v.touched)
        for i in 1:ni
          if v.touched[i]
            evaluate!(cache[i],k,$Ai,v.array[i],I.array[i])
          end
        end
      end
    end 
  end
end

@inline function add_hr_kron_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,hr_indices::Range2D,i::Integer
  )

  data = get_all_data(A)
  np,nt = size(hr_indices)
  ns = Int(size(data,1)/nt)
  for ip in 1:np
    for it in 1:nt
      ist = (it-1)*ns + i
      astp = data[ist,ip]
      data[ist,ip] = combine(astp,v)
    end
  end
  A
end

@inline function add_hr_kron_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,hr_indices::Range2D,i::Integer
  )

  data = get_all_data(A)
  np,nt = size(hr_indices)
  ns = Int(size(data,1)/nt)
  for ip in 1:np
    for it in 1:nt
      ist = (it-1)*ns + i
      ipt = (it-1)*np + ip
      astp = data[ist,ip]
      vtp = v[ipt]
      data[ist,ip] = combine(astp,vtp)
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,loc,r
  )

  for (li,i) in enumerate(is)
    ir = _indexin(i,r)
    if !isnothing(ir)
      vi = vs[li]
      add_hr_kron_entry!(combine,A,vi,loc,ir)
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,loc,r
  )

  for (li,i) in enumerate(is)
    ir = _indexin(i,r)
    if !isnothing(ir)
      get_hr_param_entry!(vi,vs,loc,li)
      add_hr_kron_entry!(combine,A,vi,loc,ir)
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vij,combine::Function,A::AbstractParamVector,vs,is,js,loc,r,c
  )

  for (lj,j) in enumerate(js)
    ic = _indexin(j,c)
    if !isnothing(ic)
      for (li,i) in enumerate(is)
        ir = _indexin(i,r)
        if !isnothing(ir)
          if ir == ic
            vij = vs[li,lj]
            add_hr_kron_entry!(combine,A,vij,loc,ir)
          end
        end
      end
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js,loc,r,c
  )

  for (lj,j) in enumerate(js)
    ic = _indexin(j,c)
    if !isnothing(ic)
      for (li,i) in enumerate(is)
        ir = _indexin(i,r)
        if !isnothing(ir)
          if ir == ic
            get_hr_param_entry!(vij,vs,loc,li,lj)
            add_hr_kron_entry!(combine,A,vij,loc,ir)
          end
        end
      end
    end
  end
  A
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,it::Int
  )

  data = get_all_data(A)
  np = param_length(A)
  for ip in 1:np
    astp = data[it,ip]
    data[it,ip] = combine(astp,v)
  end
  A
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,it::Int
  )

  data = get_all_data(A)
  np = param_length(A)
  for ip in 1:np
    ipt = (it-1)*np + ip
    vtp = v[ipt]
    astp = data[it,ip]
    data[it,ip] = combine(astp,vtp)
  end
  A
end

@inline function add_hr_lin_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,r,loc
  )

  for (ik,rk) in enumerate(r)
    li = findfirst(==(rk),is)::Int
    vi = vs[li]
    add_hr_lin_entry!(combine,A,vi,ik)
  end
  A
end

@inline function add_hr_lin_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,r,loc
  )

  for (ik,rk) in enumerate(r)
    li = findfirst(==(rk),is)::Int
    get_hr_param_entry!(vi,vs,loc,li)
    add_hr_lin_entry!(combine,A,vi,ik)
  end
  A
end

@inline function add_hr_lin_entries!(
  vij,combine::Function,A::AbstractParamVector,vs,is,js,r,c,loc
  )

  for (ik,(rk,ck)) in enumerate(zip(r,c))
    li = findfirst(==(rk),is)::Int
    lj = findfirst(==(ck),js)::Int
    vij = vs[li,lj]
    add_hr_lin_entry!(combine,A,vij,ik)
  end
  A
end

@inline function add_hr_lin_entries!(
  vij,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,js,r,c,loc
  )

  for (ik,(rk,ck)) in enumerate(zip(r,c))
    li = findfirst(==(rk),is)::Int
    lj = findfirst(==(ck),js)::Int
    get_hr_param_entry!(vij,vs,loc,li,lj)
    add_hr_lin_entry!(combine,A,vij,ik)
  end
  A
end

function RBSteady.assemble_hr_vector_add!(
  b::ArrayBlock,
  _cellvec,
  cellidsrows::ArrayBlock,
  rows::ArrayBlock,
  icells::ArrayBlock,
  locations::ArrayBlock,
  style::TransientIntegrationDomainStyle
  )

  @check cellidsrows.touched == rows.touched == icells.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(FetchBlockMap(_cellvec,i),icells[i])
      RBSteady._assemble_hr_vector_add!(b[i],cellveci,cellidsrows[i],rows[i],locations[i],style)
    end
  end
  b
end

function RBSteady.assemble_hr_vector_add!(b,_cellvec,cellidsrows,rows,icells,locations,style)
  cellvec = lazy_map(Reindex(_cellvec),icells)
  RBSteady._assemble_hr_vector_add!(b,cellvec,cellidsrows,rows,locations,style)
  b
end

function RBSteady._assemble_hr_vector_add!(b,cellvec,cellidsrows,rows,locations,style)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddTransientHREntriesMap(style,rows,locations)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    RBSteady._numeric_loop_hr_vector!(b,caches,cellvec,cellidsrows)
  end
  b
end

function RBSteady.assemble_hr_matrix_add!(
  A::ArrayBlock,
  _cellmat,
  cellidsrows::ArrayBlock,
  cellidscols::ArrayBlock,
  rows::ArrayBlock,
  cols::ArrayBlock,
  icells::ArrayBlock,
  locations::ArrayBlock,
  style::TransientIntegrationDomainStyle
  )

  @check cellidsrows.touched == cellidscols.touched == rows.touched == cols.touched == icells.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellmati = lazy_map(FetchBlockMap(_cellmat,i),icells[i])
      RBSteady._assemble_hr_matrix_add!(A[i],cellmati,cellidsrows[i],cellidscols[i],rows[i],cols[i],locations[i],style)
    end
  end
  A
end

function RBSteady.assemble_hr_matrix_add!(A,_cellmat,cellidsrows,cellidscols,rows,cols,icells,locations,style)
  cellmat = lazy_map(Reindex(_cellmat),icells)
  RBSteady._assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,rows,cols,locations,style)
  A
end

function RBSteady._assemble_hr_matrix_add!(A,cellmat,cellidsrows,cellidscols,rows,cols,locations,style)
  if length(cellmat) > 0
    rows_cache = array_cache(cellidsrows)
    cols_cache = array_cache(cellidscols)
    vals_cache = array_cache(cellmat)
    vals1 = getindex!(vals_cache,cellmat,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    cols1 = getindex!(cols_cache,cellidscols,1)
    add! = AddTransientHREntriesMap(style,(rows,cols),locations)
    add_cache = return_cache(add!,A,vals1,rows1,cols1)
    caches = add!,add_cache,vals_cache,rows_cache,cols_cache
    RBSteady._numeric_loop_hr_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
  end
  A
end