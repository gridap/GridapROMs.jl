# function _owned_integration_cells_and_idofs(interp,test,strian,trian)
#   get_cell_idofs(interp),get_owned_integration_cells(interp,strian)
# end

# function _owned_integration_cells_and_idofs(interp::GreedyInterpolation,test,strian,trian)
#   trian === strian && return get_cell_idofs(interp),get_owned_integration_cells(interp,strian)
#   domain = move_integration_domain(interp.domain,test,trian)
#   get_cell_idofs(domain),get_integration_cells(domain)
# end

# function _owned_integration_cells_and_idofs(interp,trial,test,strian,trian_moved)
#   get_cell_idofs(interp),get_owned_integration_cells(interp,strian)
# end

# function _owned_integration_cells_and_idofs(interp::GreedyInterpolation,trial,test,strian,trian_moved)
#   trian_moved === strian && return get_cell_idofs(interp),get_owned_integration_cells(interp,strian)
#   domain = move_integration_domain(interp.domain,trial,test,trian_moved)
#   get_cell_idofs(domain),get_integration_cells(domain)
# end

# function _owned_integration_cells_and_idofs(interp::BlockInterpolation{<:Interpolation,1},test,strian,trian_moved)
#   out = map(i -> _owned_integration_cells_and_idofs(interp[i],test[i],strian,trian_moved),1:length(interp))
#   map(first,out),map(last,out)
# end

# function _owned_integration_cells_and_idofs(interp::BlockInterpolation{<:Interpolation,2},trial,test,strian,trian_moved)
#   ni,nj = size(interp)
#   out = map(Iterators.product(1:ni,1:nj)) do (i,j)
#     _owned_integration_cells_and_idofs(interp[i,j],trial[j],test[i],strian,trian_moved)
#   end
#   map(first,out),map(last,out)
# end

function collect_cell_hr_matrix(trial,test,a,strian,sinterp)
  scell_mat = get_contribution(a,strian)
  cell_mat,trian = move_contributions(scell_mat,strian)
  @assert ndims(eltype(cell_mat)) == 2
  cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
  cell_mat_rc = attach_constraints_rows(test,cell_mat_c,trian)
  interp = move_integration_domain(sinterp,strian,trian)
  cell_idofs = get_cell_idofs(interp)
  cells = get_integration_cells(interp)
  (cell_mat_rc,cell_idofs,cells)
end

function collect_cell_hr_vector(test,a,strian,interp)
  scell_vec = get_contribution(a,strian)
  cell_vec,trian = move_contributions(scell_vec,strian)
  @assert ndims(eltype(cell_vec)) == 1
  cell_vec_r = attach_constraints_rows(test,cell_vec,trian)
  reindex_map = get_reindex_map(cell_vec_r)
  cell_idofs = move_idofs(get_cell_idofs(interp),reindex_map)
  cells = move_cells(get_integration_cells(interp),reindex_map)
  (cell_vec_r,cell_idofs,cells)
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

function assemble_hr_array_add!(
  A::AbstractArray{<:AbstractArray},
  cellvals,
  celldofs::AbstractArray{<:AbstractArray},
  cells::AbstractArray{<:AbstractArray}
  )

  @check size(celldofs) == size(cells) == size(A)
  for i in eachindex(celldofs)
    isempty(cells[i]) && continue
    cellvalsi = fetch_block(cellvals,i)
    assemble_hr_array_add!(A[i],cellvalsi,celldofs[i],cells[i])
  end
  A
end

function assemble_hr_array_add!(A,cellvals,celldofs,cells,args...)
  if length(celldofs) > 0
    dofs_cache = array_cache(celldofs)
    vals_cache = array_cache(cellvals)
    vals1 = getindex!(vals_cache,cellvals,1)
    rows1 = getindex!(dofs_cache,celldofs,1)
    add! = AddHREntriesMap(+,args...)
    add_cache = return_cache(add!,A,vals1,rows1)
    caches = add!,add_cache,vals_cache,dofs_cache
    _numeric_loop_hr_array!(A,caches,cellvals,celldofs,cells)
  end
  A
end

@noinline function _numeric_loop_hr_array!(arr,caches,cell_vals,cell_dofs,cells)
  add!,add_cache,vals_cache,dofs_cache = caches
  for (icell,cell) in enumerate(cells)
    dofs = getindex!(dofs_cache,cell_dofs,icell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,arr,vals,dofs)
  end
end

# utils

get_reindex_map(a) = @abstractmethod
get_reindex_map(a::LazyArray) = IdentityVector(length(a))
get_reindex_map(a::LazyArray{<:Fill{<:Reindex}}) = a.args[1].args[1]
get_reindex_map(a::LazyArray{<:Fill{<:Geometry.CombineContributionsMap}}) = a.args[1].args[1]

move_cells(cells::AbstractVector{<:Number},i::IdentityVector) = cells
move_cells(cells::AbstractVector{<:Number},i::AbstractVector) = view(cells,i)
move_cells(cells::AbstractVector{<:Number},i::Table) = view(cells,union(i))

function fetch_block(a::AbstractArray,i::Int)
  T = eltype(eltype(a))
  lazy_map(FetchBlockMap(i),T,a)
end

struct FetchBlockMap <: Map
  blockid::Int
end

function Arrays.evaluate!(c,k::FetchBlockMap,a::ArrayBlock)
  a.array[k.blockid]
end