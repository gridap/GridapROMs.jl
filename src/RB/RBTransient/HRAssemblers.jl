function RBSteady.collect_cell_hr_matrix(
  trial::FESpace,
  test::FESpace,
  a::DomainContribution,
  strian::Triangulation,
  interp::Interpolation,
  common_indices::AbstractVector)

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
  interp::Interpolation,
  common_indices::AbstractVector)

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

@inline function add_hr_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,hr_indices::Range2D,i::Integer)

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

@inline function add_hr_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,hr_indices::Range2D,i::Integer)

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

@inline function add_hr_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,ids::Tuple)

  uids = unique(ids)
  data = get_all_data(A)
  np = param_length(A)
  for it in uids
    for ip in 1:np
      astp = data[it,ip]
      data[it,ip] = combine(astp,v)
    end
  end
  A
end

@inline function add_hr_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,ids::Tuple)

  uids = unique(ids)
  data = get_all_data(A)
  np = param_length(A)
  for it in uids
    for ip in 1:np
      ipt = (it-1)*np + ip
      vtp = v[ipt]
      astp = data[it,ip]
      data[it,ip] = combine(astp,vtp)
    end
  end
  A
end

struct AddTransientHREntriesMap{A<:TransientIntegrationDomainStyle,F,I<:Range2D} <: Map
  style::A
  combine::F
  locations::I
end

function AddTransientHREntriesMap(style::TransientIntegrationDomainStyle,locations::Range2D)
  AddTransientHREntriesMap(style,+,locations)
end

get_param_time_inds(k::AddTransientHREntriesMap) = k.locations
get_param_inds(k::AddTransientHREntriesMap) = k.locations.axis1
get_time_inds(k::AddTransientHREntriesMap) = k.locations.axis2

function Arrays.return_cache(k::AddTransientHREntriesMap,A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(get_param_time_inds(k)))
end

for (T,f) in zip((:KroneckerDomain,:SequentialDomain),(:add_hr_kron_entries!,:add_hr_lin_entries!))
  @eval begin
    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,vs,is)
      $f(cache,k.combine,A,vs,is,k.locations)
    end
  end
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,loc)

  for (li,i) in enumerate(is)
    if i>0
      vi = vs[li]
      add_hr_entry!(combine,A,vi,loc,i)
    end
  end
  A
end

@inline function add_hr_kron_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,loc)

  for (li,i) in enumerate(is)
    if i>0
      get_hr_param_entry!(vi,vs,loc,li)
      add_hr_entry!(combine,A,vi,loc,i)
    end
  end
  A
end

_ipos(v::VectorValue) = any(i>0 for i in v.data)

@inline function _get_ids(v::VectorValue)
  ids = ()
  for vi in v.data
    if vi > 0
      ids = (ids...,vi)
    end
  end
  return ids
end

@inline function _get_ids(v::VectorValue,w::VectorValue)
  ids = ()
  for vi in v.data
    if vi > 0
      for wi in w.data
        if wi == vi
          ids = (ids...,wi)
          break
        end
      end
    end
  end
  return ids
end

@inline function add_hr_lin_entries!(
  vi,combine::Function,A::AbstractParamVector,vs,is,loc)

  for (li,i) in enumerate(is)
    if _ipos(i)
      vi = vs[li]
      add_hr_entry!(combine,A,vi,_get_ids(i))
    end
  end
  A
end

@inline function add_hr_lin_entries!(
  vi,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,loc)

  for (li,i) in enumerate(is)
    if _ipos(i)
      get_hr_param_entry!(vi,vs,loc,li)
      add_hr_entry!(combine,A,vi,_get_ids(i))
    end
  end
  A
end

function RBSteady.assemble_hr_array_add!(
  b::ArrayBlock,
  cellvec,
  cellidsrows::ArrayBlock,
  icells::ArrayBlock,
  locations::ArrayBlock,
  style::TransientIntegrationDomainStyle)

  @check cellidsrows.touched == icells.touched == locations.touched
  for i in eachindex(cellidsrows)
    if cellidsrows.touched[i]
      cellveci = lazy_map(BlockReindex(cellvec,i),icells.array[i])
      assemble_hr_array_add!(
        b.array[i],cellveci,cellidsrows.array[i],icells.array[i],locations.array[i],style)
    end
  end
end

function RBSteady.assemble_hr_array_add!(b,cellvec,cellidsrows,icells,locations,style)
  if length(cellvec) > 0
    rows_cache = array_cache(cellidsrows)
    vals_cache = array_cache(cellvec)
    vals1 = getindex!(vals_cache,cellvec,1)
    rows1 = getindex!(rows_cache,cellidsrows,1)
    add! = AddTransientHREntriesMap(style,locations)
    add_cache = return_cache(add!,b,vals1,rows1)
    caches = add!,add_cache,vals_cache,rows_cache
    RBSteady._numeric_loop_hr_array!(b,caches,cellvec,cellidsrows)
  end
  b
end
