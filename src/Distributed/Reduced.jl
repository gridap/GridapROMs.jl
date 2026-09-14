# trian utils

function Utils.ChildTriangulation(t::DistributedTriangulation,inds)
  models = get_background_model(t)
  trians = map(local_views(t),local_views(inds)) do t,inds
    ChildTriangulation(t,inds)
  end
  DistributedTriangulation(trians,models;metadata=t.metadata)
end

function Utils.is_parent(parent::DistributedTriangulation,child::DistributedTriangulation)
  x = map(local_views(parent),local_views(child)) do parent,child
    Utils.is_parent(parent,child)
  end
  reduce(&,x)
end

# reduced basis spaces

const DistributedRBSpace{S<:DistributedFESpace} = RBSpace{S}

const DistributedSingleFieldRBSpace{S<:DistributedSingleFieldFESpace} = DistributedRBSpace{S}
const DistributedMultiFieldRBSpace{S<:DistributedMultiFieldFESpace} = DistributedRBSpace{S}

function GridapDistributed.local_views(r::DistributedRBSpace)
  map(local_views(r.space),local_views(r.subspace)) do space,subspace
    RBSpace(space,subspace)
  end
end

for T in (:DistributedSingleFieldFESpace,:DistributedMultiFieldFESpace)
  @eval begin
    function FESpaces.FEFunction(f::$T,fv::RBParamVector,args...)
      FEFunction(f,fv.fe_data,args...)
    end

    function FESpaces.EvaluationFunction(f::$T,fv::RBParamVector,args...)
      EvaluationFunction(f,fv.fe_data,args...)
    end
  end
end

function RBSteady._convert_to_block(V::DistributedMultiFieldFESpace)
  part_fe_space = map(local_views(V)) do space
    RBSteady._convert_to_block(space)
  end
  DistributedMultiFieldFESpace(V.field_fe_space,part_fe_space,V.gids,V.vector_type)
end

function Base.getindex(r::DistributedMultiFieldRBSpace,i::Integer)
  mfe = get_fe_space(r)
  rsp = get_reduced_subspace(r)
  return reduced_subspace(mfe.field_fe_space[i],rsp[i])
end

function Base.iterate(r::DistributedMultiFieldRBSpace,state=1)
  if state > num_fields(r)
    return nothing
  end
  mfe = get_fe_space(r)
  rsp = get_reduced_subspace(r)
  ri = reduced_subspace(mfe.field_fe_space[state],rsp[state])
  return ri,state+1
end

MultiField.MultiFieldStyle(r::DistributedMultiFieldRBSpace) = MultiFieldStyle(get_fe_space(r))
MultiField.num_fields(r::DistributedMultiFieldRBSpace) = num_fields(get_fe_space(r))
Base.length(r::DistributedMultiFieldRBSpace) = num_fields(r)

# integration domains

struct LocalDEIMIndices{Tr,Tc,A<:AbstractLocalIndices} <: AbstractVector{Tr}
  global_rows::Vector{Tr}
  global_cols::Vector{Tc}
  index_parts::A
end

LocalDEIMIndices(index_parts) = LocalDEIMIndices(Int[],Int[],index_parts)

Base.size(a::LocalDEIMIndices) = size(a.global_rows)
Base.IndexStyle(::Type{<:LocalDEIMIndices}) = IndexLinear()
Base.getindex(a::LocalDEIMIndices,i::Int) = getindex(a.global_rows,i)
Base.setindex!(a::LocalDEIMIndices,v,i::Int) = setindex!(a.global_rows,v,i)
Base.copy(a::LocalDEIMIndices) = LocalDEIMIndices(copy(a.global_rows),copy(a.global_cols),a.index_parts)

function RBSteady._evaluate!(a,cellrows,rows::LocalDEIMIndices)
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

function RBSteady._evaluate!(a,cellrows,cellcols,rows::LocalDEIMIndices,cols::LocalDEIMIndices)
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

function RBSteady.DEIM(basis::GenericPMatrix)
  T = eltype(basis)
  n = size(basis,2)
  I = zeros(Int,n)
  parts = partition(axes(basis,1))
  Iparts = map(LocalDEIMIndices,parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = findrow(res)
  _push_parts!(Iparts,I,1)
  _from_submatrix!(basisI,basis,I,1)
  for l = 2:n
    PᵀU = view(basisI,1:l-1,1:l-1)
    Pᵀuₗ = view(basisI,1:l-1,l)
    c = vec(PᵀU \ Pᵀuₗ)
    map(own_values(res),own_values(basis)) do ro,bo
      @. ro = bo[:,l]
      mul!(ro,view(bo,:,1:l-1),c,-1.0,1.0)
    end
    I[l] = findrow(res)
    _push_parts!(Iparts,I,l)
    _from_submatrix!(basisI,basis,I,l)
  end
  return Iparts,basisI
end

function RBSteady.SOPT(basis::GenericPMatrix)
  T = eltype(basis)
  n = size(basis,2)
  I = zeros(Int,n)  
  parts = partition(axes(basis,1))
  Iparts = map(LocalDEIMIndices,parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = findrow(res)
  _push_parts!(Iparts,I,1)
  _from_submatrix!(basisI,basis,I,1)
  for l in 2:n
    P = I[1:l-1]
    PᵀU = view(basisI,1:l-1,1:l)
    G = PᵀU'*PᵀU
    colnorms2 = vec(sum(abs2,PᵀU;dims=1))
    Il = _best_s_opt_index(basis,P,G,colnorms2,l)
    @check Il > 0
    I[l] = Il
    _push_parts!(Iparts,I,l)
    _from_submatrix!(basisI,basis,I,l)
  end
  return Iparts,basisI
end

for f in (:DEIM,:SOPT)
  @eval begin
    function RBSteady.$f(A::PSparseMatrix)
      B = get_all_data(A)
      I,AI = $f(B)
      n = size(AI,1)
      r,c = map(local_views(I),local_values(A),flat_row_partition(B)) do I,A,rci
        rcache = zeros(Int,n)
        ccache = zeros(Int,n)
        _remap!(I,global_to_local(rci))
        r,c = recast_split_indices(I,testitem(A))
        _remap!(r,local_to_global(row_partition(rci)))
        _remap!(c,local_to_global(col_partition(rci)))
        for (k,sk) in enumerate(r.global_cols)
          rcache[sk] = r[k]
          ccache[sk] = c[k]
        end
        (rcache,ccache)
      end |> tuple_of_arrays
      # assemble the full per-slot (global row & col dof) across ranks
      op(a,b) = max.(a,b) # assign a DEIM index to only one rank, though it may appear on multiple ranks
      grows = reduce(op,r)
      gcols = reduce(op,c)
      # per rank: keep every slot whose row AND col dof is local
      R′,C′ = map(flat_row_partition(B)) do rci
        g2lr = global_to_local(row_partition(rci))
        g2lc = global_to_local(col_partition(rci))
        ikeep,rkeep,ckeep = _keep_rows_and_cols(grows,gcols,g2lr,g2lc)
        R′ = LocalDEIMIndices(rkeep,copy(ikeep),row_partition(rci))
        C′ = LocalDEIMIndices(ckeep,copy(ikeep),col_partition(rci))
        (R′,C′)
      end |> tuple_of_arrays
      return (R′,C′),AI
    end
  end
end

for T in (:AbstractSparseMatrix,:SubSparseMatrix)
  @eval begin
    function DofMaps.recast_split_indices(sids::LocalDEIMIndices,a::$T)
      rids,cids = recast_split_indices(sids.global_rows,a)
      r = LocalDEIMIndices(rids,copy(sids.global_cols),sids.index_parts)
      c = LocalDEIMIndices(cids,copy(sids.global_cols),sids.index_parts)
      (r,c)
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

struct DistributedIntegrationDomain{A} <: IntegrationDomain
  domains::A
end

GridapDistributed.local_views(a::DistributedIntegrationDomain) = local_views(a.domains)

for f in (:get_integration_cells,:get_cell_idofs,:get_interpolation_dofs)
  @eval begin
    function RBSteady.$f(a::DistributedIntegrationDomain)
      map(local_views(a)) do a
        $f(a)
      end
    end
  end
end

function RBSteady.IntegrationDomain(
  trian::DistributedTriangulation,
  test::DistributedRBSpace,
  rows::AbstractArray{<:AbstractVector}
  )

  gids = get_free_dof_ids(test)
  domains = map(
    local_views(trian),
    local_views(test),
    local_views(rows),
    local_views(gids)
    ) do trian,test,rows,gids
    lrows = _remap(rows,global_to_local(gids))
    domain = IntegrationDomain(trian,test,lrows)
    grows = _remap(lrows,local_to_global(gids))
    GenericDomain(get_integration_cells(domain),get_cell_idofs(domain),grows)
  end
  DistributedIntegrationDomain(domains)
end

function RBSteady.IntegrationDomain(
  trian::DistributedTriangulation,
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  rows::AbstractArray{<:AbstractVector},
  cols::AbstractArray{<:AbstractVector}
  )

  cgids = get_free_dof_ids(trial)
  rgids = get_free_dof_ids(test)
  domains = map(
    local_views(trian),
    local_views(trial),
    local_views(test),
    local_views(rows),
    local_views(cols),
    local_views(rgids),
    local_views(cgids)
    ) do trian,trial,test,rows,cols,rgids,cgids
    lrows = _remap(rows,global_to_local(rgids))
    lcols = _remap(cols,global_to_local(cgids))
    domain = IntegrationDomain(trian,trial,test,lrows,lcols)
    grows = _remap(lrows,local_to_global(rgids))
    gcols = _remap(lcols,local_to_global(cgids))
    GenericDomain(get_integration_cells(domain),get_cell_idofs(domain),(grows,gcols))
  end
  DistributedIntegrationDomain(domains)
end

# hyper-reduction

struct DistributedInterpolation{A} <: Interpolation
  interps::A
end

function RBSteady.Interpolation(red::NoHyperReduction,trian::DistributedTriangulation)
  interps = map(local_views(trian)) do ti
    Interpolation(red,ti)
  end
  DistributedInterpolation(interps)
end

function RBSteady.GreedyInterpolation(interp,domain::DistributedIntegrationDomain)
  interps = map(local_views(domain)) do domain
    GreedyInterpolation(interp,domain)
  end
  DistributedInterpolation(interps)
end

GridapDistributed.local_views(a::DistributedInterpolation) = local_views(a.interps)

for f in (:get_integration_cells,:get_cell_idofs,:get_interpolation_dofs)
  @eval begin
    function RBSteady.$f(a::DistributedInterpolation)
      map(local_views(a)) do a
        $f(a)
      end
    end
  end
end

function FESpaces.interpolate!(
  cache::AbstractArray{<:AbstractArray},
  a::DistributedInterpolation,
  b::AbstractArray{<:AbstractArray}
  )

  map(local_views(cache),local_views(a),local_views(b)) do cache,interp,b
    interpolate!(cache,interp,b)
  end
end

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::DistributedInterpolation)
  red_cells = get_integration_cells(a)
  trians = map(local_views(trian),local_views(red_cells)) do ti,ci
    ChildTriangulation(ti,ci)
  end
  model = get_background_model(trian)
  DistributedTriangulation(trians,model)
end

function RBSteady.get_at_domain(s::DistributedSparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  inds = map(local_values(s),local_views(rows),local_views(cols)) do s,rows,cols
    @check rows.global_cols == cols.global_cols
    if !isempty(rows)
      sparsity = get_sparsity(get_dof_map(s))
      rc = sparsify_split_indices(rows,cols,sparsity)
      LocalDEIMIndices(rc,rows.global_cols,rows.index_parts)
    else
      LocalDEIMIndices(rows.index_parts)
    end
  end
  get_at_domain(s.snaps,inds)
end

function RBSteady.get_at_domain(a::GenericPArray,rows::AbstractArray{<:LocalDEIMIndices})
  n = size(a,2)
  @check reduce(+,map(length,rows)) == n
  datav = zeros(eltype(a),n,n)
  map(local_values(a),local_views(rows)) do data,rows
    g2l = global_to_local(rows.index_parts)
    if !isempty(rows.global_rows)
      for (gri,i) in zip(rows.global_rows,rows.global_cols)
        lri = g2l[gri]
        for k in axes(data,2)
          datav[i,k] = data[lri,k]
        end
      end
    end
  end
  ConsecutiveParamArray(datav)
end

struct DistributedHRProjection{A,B} <: HRProjection{A,B}
  basis::A
  style::B
  interpolation::DistributedInterpolation
end

function RBSteady.HRProjection(basis::ReducedProjection,style::HyperReduction,interp::DistributedInterpolation)
  DistributedHRProjection(basis,style,interp)
end

function GridapDistributed.local_views(a::DistributedHRProjection)
  map(local_views(a.interpolation)) do interp
    HRProjection(a.basis,a.style,interp)
  end
end

RBSteady.get_basis(a::DistributedHRProjection) = a.basis
RBSteady.get_style(a::DistributedHRProjection) = a.style
RBSteady.get_interpolation(a::DistributedHRProjection) = a.interpolation

function FESpaces.interpolate!(
  b̂::AbstractArray,
  _coeff::AbstractArray{<:AbstractArray},
  a::DistributedHRProjection,
  x::AbstractArray{<:AbstractArray}
  )

  o = one(eltype2(b̂))
  interpolate!(_coeff,get_interpolation(a),x)
  coeff = sreduce(_coeff)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

function FESpaces.interpolate!(
  b̂::AbstractArray,
  _coeff::AbstractArray{<:AbstractArray},
  a::DistributedHRProjection{A,NoHyperReduction} where A,
  x::AbstractArray{<:AbstractArray}
  )

  coeff = sreduce(_coeff)
  o = one(eltype2(b̂))
  axpy!(o,coeff,b̂)
  return b̂
end

function RBSteady.reduced_triangulation(trian::DistributedTriangulation,a::DistributedHRProjection)
  reduced_triangulation(trian,get_interpolation(a))
end

function Base.fill!(a::AbstractArray{<:AbstractParamArray},b::Number)
  map(local_views(a)) do a
    fill!(a,b)
  end
  a
end

function RBSteady.allocate_coefficient(a::DistributedHRProjection)
  map(local_views(a)) do a
    RBSteady.allocate_coefficient(a)
  end
end

function RBSteady.allocate_coefficient(a::DistributedHRProjection,r::AbstractRealisation)
  map(local_views(a)) do a
    RBSteady.allocate_coefficient(a,r)
  end
end

function RBSteady.collect_cell_hr_matrix(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  a::DistributedDomainContribution,
  strian::DistributedTriangulation,
  interp::DistributedInterpolation,
  args...
  )

  map(
    local_views(trial),
    local_views(test),
    local_views(a),
    local_views(strian),
    local_views(interp)
    ) do trial,test,a,strian,interp
    collect_cell_hr_matrix(trial,test,a,strian,interp,args...)
  end
end

function RBSteady.collect_cell_hr_vector(
  test::DistributedRBSpace,
  a::DistributedDomainContribution,
  strian::DistributedTriangulation,
  interp::DistributedInterpolation,
  args...
  )

  map(
    local_views(test),
    local_views(a),
    local_views(strian),
    local_views(interp)
    ) do test,a,strian,interp
    collect_cell_hr_vector(test,a,strian,interp,args...)
  end
end

function RBSteady.assemble_hr_array_add!(A::AbstractArray{<:AbstractArray},celldata::AbstractArray{<:Tuple})
  map(local_views(A),local_views(celldata)) do A,celldata
    assemble_hr_array_add!(A,celldata)
  end
end

for T in (:GenericPMatrix,:DistributedSnapshots)
  @eval begin
    function Utils.induced_norm(a::$T)
      _norm_part(x) = induced_norm(x)^2
      n = reduce(+,map(_norm_part,own_values(a)))
      sqrt(n)
    end
  end
end

# utils

function _subfill!(a::AbstractVector,b::AbstractVector,ia,ib)
  a[ia] = b[ib]
end

function _subfill!(a::AbstractMatrix,b::AbstractMatrix,ia,ib)
  @check size(a,2) == size(b,2)
  @inbounds for k in axes(a,2)
    a[ia,k] = b[ib,k]
  end
end

function _from_submatrix!(aI,a,I,l)
  aI .+= map(own_values(a),partition(axes(a,1))) do oa,ra
    g2o = global_to_own(ra)
    c = similar(aI)
    fill!(c,zero(eltype(c)))
    for k in l
      or = g2o[I[k]]
      or > 0 && _subfill!(c,oa,k,or)
    end
    c
  end |> sreduce
end

function _push_parts!(a::AbstractArray{<:LocalDEIMIndices},I,l)
  gl = I[l]
  map(a) do a
    if global_to_local(a.index_parts)[gl] > 0
      push!(a.global_rows,gl)
      push!(a.global_cols,l)
    end
  end
end

function _remap!(x,x_to_y)
  for (i,xi) in enumerate(x)
    x[i] = x_to_y[xi]
  end
end

function _remap(x,x_to_y)
  x′ = copy(x)
  _remap!(x′, x_to_y)
  x′
end

function _keep_rows_and_cols(rows,cols,rowmap,colmap)
  @check length(rows) == length(cols)
  @check length(rowmap) == length(colmap)
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

function _best_s_opt_index(basis::GenericPMatrix,P,G,colnorms2,l)
  best_pairs = map(own_values(basis),partition(axes(basis,1))) do bo,ra
    best_logS = -Inf
    best_gi = 0
    for oi in axes(bo,1)
      gi = own_to_global(ra)[oi]
      gi ∈ P && continue
      q = view(bo,oi,1:l)
      logdet_plus = RBSteady.robust_logdet(G + q*q')
      colnorms2_plus = colnorms2 .+ abs2.(q)
      logS = (0.5/l)*(logdet_plus - sum(log,colnorms2_plus))
      if logS > best_logS
        best_logS = logS
        best_gi = gi
      end
    end
    best_logS => best_gi
  end
  return second(reduce(max,best_pairs,init=(-Inf=>0)))
end

function RBSteady._setup(U::DistributedMultiFieldRBSpace,u0::PVector)
  map(local_views(U),local_values(u0)) do U,u0
    RBSteady._setup(U,u0)
  end |> mortar
end