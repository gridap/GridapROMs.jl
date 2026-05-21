function RBSteady.collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::TransientIntegrationDomain,
  common_indices::AbstractVector
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  locations = get_param_itimes(interp,common_indices)
  style = get_domain_style(interp)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_idofs,icells,locations,style)
end

function RBSteady.collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::TransientIntegrationDomain,
  common_indices::AbstractVector
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  locations = get_param_itimes(interp,common_indices)
  style = get_domain_style(interp)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_idofs,icells,locations,style)
end

function RBSteady.collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::TransientIntegrationDomain{<:SequentialDomain},
  common_indices::AbstractVector
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  rows,cols = get_interpolation_dofs(interp)
  iurowcol_to_irowcol = get_iurowcol_to_irowcol(rows,cols)
  style = get_domain_style(interp)

  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  (cell_mat_rc,cell_idofs,icells,iurowcol_to_irowcol,style)
end

function RBSteady.collect_cell_hr_vector(
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::TransientIntegrationDomain{<:SequentialDomain},
  common_indices::AbstractVector
  )

  cell_idofs = get_cell_idofs(interp)
  icells = get_owned_icells(interp,strian)
  rows = get_interpolation_dofs(interp)
  iurow_to_irow = get_iurow_to_irow(rows)
  style = get_domain_style(interp)

  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  (cell_vec_r,cell_idofs,icells,iurow_to_irow,style)
end

struct AddTransientHREntriesMap{A<:TransientIntegrationDomainStyle,F,I} <: Map
  style::A
  combine::F
  locations::I
end

function AddTransientHREntriesMap(style::TransientIntegrationDomainStyle,locations)
  AddTransientHREntriesMap(style,+,locations)
end

function Arrays.return_cache(k::AddTransientHREntriesMap{KroneckerDomain},A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(k.locations))
end

function Arrays.return_cache(k::AddTransientHREntriesMap{SequentialDomain},A,vs::ParamBlock,args...)
  cv = zeros(eltype2(vs),param_length(A))
  cl = array_cache(k.locations)
  (cv,cl)
end

for (T,f) in zip((:KroneckerDomain,:SequentialDomain),(:add_hr_kron_entries!,:add_hr_lin_entries!))
  @eval begin
    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},b,vs,is)
      $f(cache,k.combine,b,vs,is,k.locations)
    end
  end
end

for T in (:KroneckerDomain,:SequentialDomain)
  @eval begin
    function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A,v::MatrixBlock,IJ::MatrixBlock)
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

    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,v::MatrixBlock,IJ::MatrixBlock)
      ni,nj = size(v.touched)
      for j in 1:nj
        for i in 1:ni
          if v.touched[i,j]
            evaluate!(cache[i,j],k,A,v.array[i,j],IJ.array[i,j])
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
      function Arrays.return_cache(k::AddTransientHREntriesMap{$T},A::$MT,v::MatrixBlock,IJ::MatrixBlock)
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

      function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A::$MT,v::MatrixBlock,IJ::MatrixBlock)
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

function get_hr_kron_param_entry!(v::AbstractVector,b::GenericParamBlock,hr_indices,i...)
  for (k,hrk) in enumerate(hr_indices)
    v[k] = b.data[hrk][i...]
  end
  v
end

function get_hr_kron_param_entry!(v::AbstractVector,b::TrivialParamBlock,hr_indices,i...)
  vk = b.data[i...]
  fill!(v,vk)
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,loc
  )

  for (li,i) in enumerate(is)
    if i>0
      vi = vs[li]
      add_hr_kron_entry!(combine,A,vi,loc,i)
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,loc
  )

  for (li,i) in enumerate(is)
    if i>0
      get_hr_kron_param_entry!(vi,vs,loc,li)
      add_hr_kron_entry!(combine,A,vi,loc,i)
    end
  end
  A
end

function get_hr_lin_param_entry!(v::AbstractVector,b::GenericParamBlock,i::Integer)
  for ip in eachindex(v)
    @inbounds v[ip] = b.data[ip][i]
  end
  v
end

function get_hr_lin_param_entry!(v::AbstractVector,b::TrivialParamBlock,i::Integer)
  fill!(v,b.data[i])
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,k::Integer
  )

  data = get_all_data(A)
  np = param_length(A)
  for ip in 1:np
    data[k,ip] = combine(data[k,ip],v)
  end
  A
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,k::Integer
  )

  data = get_all_data(A)
  np = param_length(A)
  for ip in 1:np
    data[k,ip] = combine(data[k,ip],v[ip])
  end
  A
end

@inline function add_hr_lin_entries!(
  cache,combine::Function,A::AbstractParamVector,vs,is,iurow_to_irow::Table
  )

  vi,ucache = cache
  for (li,i) in enumerate(is)
    if i > 0
      vi = vs[li]
      ks = getindex!(ucache,iurow_to_irow,i)
      for k in ks
        add_hr_lin_entry!(combine,A,vi,k)
      end
    end
  end
  A
end

@inline function add_hr_lin_entries!(
  cache,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,iurow_to_irow::Table
  )

  vi,ucache = cache
  for (li,i) in enumerate(is)
    if i > 0
      get_hr_lin_param_entry!(vi,vs,li)
      ks = getindex!(ucache,iurow_to_irow,i)
      for k in ks
        add_hr_lin_entry!(combine,A,vi,k)
      end
    end
  end
  A
end

function RBSteady.assemble_hr_array_add!(
  b::ArrayBlock,
  _cellvals,
  celldofs::ArrayBlock,
  icells::ArrayBlock,
  locations::ArrayBlock,
  style::TransientIntegrationDomainStyle
  )

  @check celldofs.touched == icells.touched == locations.touched
  for i in eachindex(celldofs)
    if celldofs.touched[i]
      cellvalsi = lazy_map(FetchBlockMap(_cellvals,i),icells[i])
      RBSteady._assemble_hr_array_add!(b[i],cellvalsi,celldofs[i],locations[i],style)
    end
  end
  b
end

function RBSteady.assemble_hr_array_add!(b,_cellvals,celldofs,icells,locations,style)
  cellvals = lazy_map(Reindex(_cellvals),icells)
  RBSteady._assemble_hr_array_add!(b,cellvals,celldofs,locations,style)
  b
end

function RBSteady._assemble_hr_array_add!(b,cellvals,celldofs,locations,style)
  if length(cellvals) > 0
    dofs_cache = array_cache(celldofs)
    vals_cache = array_cache(cellvals)
    vals1 = getindex!(vals_cache,cellvals,1)
    dofs1 = getindex!(dofs_cache,celldofs,1)
    add! = AddTransientHREntriesMap(style,locations)
    add_cache = return_cache(add!,b,vals1,dofs1)
    caches = add!,add_cache,vals_cache,dofs_cache
    RBSteady._numeric_loop_hr_array!(b,caches,cellvals,celldofs)
  end
  b
end