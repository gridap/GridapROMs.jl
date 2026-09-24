function RBSteady.collect_cell_hr_matrix(trial,test,a,strian,interp,common_indices)
  cell_mat_rc,cell_idofs,cells = collect_cell_hr_matrix(trial,test,a,strian,interp)
  locations = get_locations(interp,common_indices)
  style = get_interpolation_style(interp)
  (cell_mat_rc,cell_idofs,cells,locations,style)
end

function RBSteady.collect_cell_hr_vector(test,a,strian,interp,common_indices)
  cell_vec_r,cell_idofs,cells = collect_cell_hr_vector(test,a,strian,interp)
  locations = get_locations(interp,common_indices)
  style = get_interpolation_style(interp)
  (cell_vec_r,cell_idofs,cells,locations,style)
end

function get_hr_param_entry!(v::AbstractVector,A::GenericParamBlock,hr_indices,i...)
  for (k,hrk) in enumerate(hr_indices)
    v[k] = A.data[hrk][i...]
  end
  v
end

function get_hr_param_entry!(v::AbstractVector,A::TrivialParamBlock,hr_indices,i...)
  vk = A.data[i...]
  fill!(v,vk)
end

struct AddTransientHREntriesMap{A<:InterpolationStyle,F,I} <: Map
  style::A
  combine::F
  locations::I
end

function RBSteady.AddHREntriesMap(combine::Function,locations,style::InterpolationStyle)
  AddTransientHREntriesMap(style,combine,locations)
end

function Arrays.return_cache(k::AddTransientHREntriesMap{KroneckerStyle},A,vs::ParamBlock,args...)
  zeros(eltype2(vs),length(k.locations))
end

function Arrays.return_cache(k::AddTransientHREntriesMap{SequentialStyle},A,vs,args...)
  sloc,tloc = k.locations
  array_cache(sloc)
end

function Arrays.return_cache(k::AddTransientHREntriesMap{SequentialStyle},A,vs::ParamBlock,args...)
  sloc,tloc = k.locations
  cv = zeros(eltype2(vs),length(tloc))
  cl = array_cache(sloc)
  (cv,cl)
end

for (T,f) in zip((:KroneckerStyle,:SequentialStyle),(:add_hr_kron_entries!,:add_hr_lin_entries!))
  @eval begin
    function Arrays.evaluate!(cache,k::AddTransientHREntriesMap{$T},A,vs,is)
      $f(cache,k.combine,A,vs,is,k.locations)
    end
  end
end

for T in (:KroneckerStyle,:SequentialStyle)
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
  nt == 0 && return A # e.g. a structurally-empty (all-zero) jacobian block: no time indices to add
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
  nt == 0 && return A # e.g. a structurally-empty (all-zero) jacobian block: no time indices to add
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
      get_hr_param_entry!(vi,vs,loc,li)
      add_hr_kron_entry!(combine,A,vi,loc,i)
    end
  end
  A
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::Number,ids
  )

  data = get_all_data(A)
  np = param_length(A)
  for it in ids
    for ip in 1:np
      astp = data[it,ip]
      data[it,ip] = combine(astp,v)
    end
  end
  A
end

@inline function add_hr_lin_entry!(
  combine::Function,A::ConsecutiveParamVector,v::AbstractVector,ids
  )

  data = get_all_data(A)
  np = param_length(A)
  for it in ids
    for ip in 1:np
      ipt = (it-1)*np + ip
      vtp = v[ipt]
      astp = data[it,ip]
      data[it,ip] = combine(astp,vtp)
    end
  end
  A
end

@inline function add_hr_lin_entries!(
  cache,combine::Function,A::AbstractParamVector,vs,is,loc
  )

  sloc,tloc = loc
  for (li,i) in enumerate(is)
    if i > 0
      vi = vs[li]
      ks = getindex!(cache,sloc,i)
      add_hr_lin_entry!(combine,A,vi,ks)
    end
  end
  A
end

@inline function add_hr_lin_entries!(
  cache,combine::Function,A::AbstractParamVector,vs::ParamBlock,is,loc
  )

  sloc,tloc = loc
  vi,scache = cache
  for (li,i) in enumerate(is)
    if i > 0
      get_hr_param_entry!(vi,vs,tloc,li)
      ks = getindex!(scache,sloc,i)
      for k in ks
        add_hr_lin_entry!(combine,A,vi,k)
      end
    end
  end
  A
end

function RBSteady.assemble_hr_array_add!(
  A::AbstractArray{<:AbstractArray},
  cellvals,
  celldofs::AbstractArray{<:AbstractArray},
  cells::AbstractArray{<:AbstractArray},
  locations::AbstractArray{<:AbstractArray},
  style::InterpolationStyle
  )

  @check size(celldofs) == size(cells) == size(locations) == size(A)
  for i in eachindex(celldofs)
    cellvalsi = fetch_block(cellvals,i)
    assemble_hr_array_add!(A[i],cellvalsi,celldofs[i],cells[i],locations[i],style)
  end
  A
end