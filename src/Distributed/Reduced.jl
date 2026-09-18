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

# galerkin projections

function RBSteady.galerkin_projection(Φl::GenericPMatrix,b::PVector)
  map(own_values(Φl),own_values(b)) do Φlo,bo
    galerkin_projection(Φlo,bo)
  end |> sreduce
end

function RBSteady.galerkin_projection(Φl::GenericPMatrix,A::PSparseMatrix,Φr::GenericPMatrix)
  TS = promote_type(eltype(Φl),eltype(Φr))
  nleft = size(Φl,2)
  n = getany(map(param_length,partition(A)))
  nright = size(Φr,2)
  Â = zeros(TS,nleft,nright,n)
  _galerkin_mul!(Â,Φl,A,Φr)
  return Â
end

# integration domains

function RBSteady.DEIM(basis::GenericPMatrix)
  T = eltype(basis)
  m,n = size(basis)
  parts = partition(axes(basis,1))
  (m == 0 || n == 0) && return map(LocalDofs,parts),zeros(T,0,0)
  I = zeros(Int,n)
  Iparts = map(LocalDofs,parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = findrow(res)
  _fill_parts!(Iparts,basisI,basis,I,1)
  for l = 2:n
    PᵀU = view(basisI,1:l-1,1:l-1)
    Pᵀuₗ = view(basisI,1:l-1,l)
    c = vec(PᵀU \ Pᵀuₗ)
    map(own_values(res),own_values(basis)) do ro,bo
      @. ro = bo[:,l]
      mul!(ro,view(bo,:,1:l-1),c,-1.0,1.0)
    end
    I[l] = findrow(res)
    _fill_parts!(Iparts,basisI,basis,I,l)
  end
  return Iparts,basisI
end

function RBSteady.SOPT(basis::GenericPMatrix)
  T = eltype(basis)
  m,n = size(basis)
  parts = partition(axes(basis,1))
  (m == 0 || n == 0) && return map(LocalDofs,parts),zeros(T,0,0)
  I = zeros(Int,n)
  Iparts = map(LocalDofs,parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = findrow(res)
  _fill_parts!(Iparts,basisI,basis,I,1)
  for l in 2:n
    P = I[1:l-1]
    PᵀU = view(basisI,1:l-1,1:l)
    G = PᵀU'*PᵀU
    colnorms2 = vec(sum(abs2,PᵀU;dims=1))
    Il = _best_s_opt_index(basis,P,G,colnorms2,l)
    @check Il > 0
    I[l] = Il
    _fill_parts!(Iparts,basisI,basis,I,l)
  end
  return Iparts,basisI
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

const TransientDistributedIntegrationDomain{A<:TransientIntegrationDomainStyle,I<:DistributedIntegrationDomain,Ti<:Integer} = TransientIntegrationDomain{A,I,Ti}

function GridapDistributed.local_views(a::TransientDistributedIntegrationDomain)
  map(local_views(a.domain_space)) do domain_space
    TransientIntegrationDomain(a.domain_style,domain_space,a.indices_time)
  end
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

function RBSteady.GreedyInterpolation(interp,domain::TransientDistributedIntegrationDomain)
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

for f in (:get_domain_style,:get_indices_time)
  @eval RBTransient.$f(a::DistributedInterpolation) = $f(getany(a.interps))
end

for f in (:get_itimes,:get_locations)
  @eval RBTransient.$f(a::DistributedInterpolation,ids) = $f(getany(a.interps),ids)
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

function RBSteady.get_at_domain(s::DistributedSparseSnapshots,rowscols::Tuple)
  rows,cols = rowscols
  inds = map(local_values(s),local_views(rows),local_views(cols)) do s,rows,cols
    @check rows.global_cols == cols.global_cols
    if !isempty(rows)
      dof_map = get_dof_map(s)
      rc = sparsify_split_indices(rows,cols,dof_map)
      LocalDofs(rc,rows.global_cols,rows.index_parts)
    else
      LocalDofs(rows.index_parts)
    end
  end
  get_at_domain(s.snaps,inds)
end

function RBSteady.get_at_domain(s::DistributedSnapshots,rows::AbstractArray{<:LocalDofs})
  n = size(s,2)
  @check reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows)) == n
  datav = map(local_values(s),local_views(rows)) do data,rows
    x = zeros(eltype(data),n,n)
    g2l = global_to_local(rows.index_parts)
    if !isempty(rows.global_rows)
      for (gri,i) in zip(rows.global_rows,rows.global_cols)
        lri = g2l[gri]
        for k in axes(data,2)
          x[i,k] = data[lri,k]
        end
      end
    end
    x
  end |> nzreduce
  ConsecutiveParamArray(datav)
end

# for f in (:get_at_kron_domain,:get_at_seq_domain)
#   @eval begin
#     function RBTransient.$f(
#       s::DistributedTransientSparseSnapshots,
#       rowscols::Tuple,
#       indices_time::AbstractVector{<:Integer}
#       )

#       rows,cols = rowscols
#       inds = map(local_values(s),local_views(rows),local_views(cols)) do s,rows,cols
#         @check rows.global_cols == cols.global_cols
#         if !isempty(rows)
#           dof_map = get_dof_map(s)
#           rc = sparsify_split_indices(rows,cols,dof_map)
#           LocalDofs(rc,rows.global_cols,rows.index_parts)
#         else
#           LocalDofs(rows.index_parts)
#         end
#       end
#       RBTransient.$f(s,inds,indices_time)
#     end
#   end
# end

function RBTransient.get_at_kron_domain(
  s::DistributedTransientSnapshots,
  rows::AbstractArray{<:LocalDofs},
  indices_time::AbstractVector{<:Integer}
  )

  ns = reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows))
  nt = length(indices_time)
  np = num_params(s)
  datav = map(local_values(s),local_views(rows)) do s,rows
    data = flatten(s)
    x = zeros(eltype(data),ns*nt,np)
    g2l = global_to_local(rows.index_parts)
    if !isempty(rows.global_rows)
      for (j,itime) in enumerate(indices_time)
        for (gri,i) in zip(rows.global_rows,rows.global_cols)
          lri = g2l[gri]
          for k in axes(data,2)
            x[(j-1)*ns+i,k] = data[lri,k,itime]
          end
        end
      end
    end
    x
  end |> nzreduce
  ConsecutiveParamArray(datav)
end

function RBTransient.get_at_seq_domain(
  s::DistributedTransientSnapshots,
  rows::AbstractArray{<:LocalDofs},
  indices_time::AbstractVector{<:Integer}
  )

  n = length(indices_time)
  np = num_params(s)
  @check reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows)) == n
  datav = map(local_values(s),local_views(rows)) do s,rows
    data = flatten(s)
    x = zeros(eltype(data),n,np)
    for i in CartesianIndices(x)
      x[i] = data[rows[i.I[1]],i.I[2],indices_time[i.I[1]]]
    end
    x
  end |> nzreduce
  ConsecutiveParamArray(datav)
end

const DistributedHRProjection{A<:HyperReduction,B<:Projection,C<:DistributedInterpolation} = RBSteady.GenericHRProjection{A,B,C}

function GridapDistributed.local_views(a::DistributedHRProjection)
  map(local_views(a.interpolation)) do interp
    HRProjection(a.basis,a.style,interp)
  end
end

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
  a::DistributedHRProjection{NoHyperReduction,B} where B,
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

# norm utils 

for T in (:GenericPMatrix,:GenericPArray,:DistributedSnapshots)
  @eval begin
    function Utils.induced_norm(a::$T)
      _norm_part(x) = induced_norm(x)^2
      n = reduce(+,map(_norm_part,own_values(a)))
      sqrt(n)
    end
  end
end

for T in (:GenericPArray,:DistributedSnapshots)
  @eval begin
    function Utils.induced_norm(a::$T,norm_matrix::AbstractMatrix)
      values = map(local_values(a)) do a 
        reshape(a,size(a,1),:)
      end
      a′ = GenericPArray(values,partition(axes(a,1)))
      sqrtabs(mean(diag(a′'*(norm_matrix*a′))))
    end
  end
end

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

# generic utils

function _galerkin_mul!(
  d::AbstractArray{<:Number,3},
  c::GenericPArray,
  a::PSparseMatrix,
  b::GenericPArray
  )

  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(c,1),axes(a,1))
  @boundscheck @assert PartitionedArrays.matching_own_indices(axes(a,2),axes(b,1))
  if !PartitionedArrays.matching_ghost_indices(axes(a,2),axes(b,1))
    b = _change_layout(b,partition(axes(a,2)))
  end
  # Start the exchange
  t = consistent!(b)
  # Meanwhile, process the owned block into a per-rank local buffer.
  ld = map(own_values(c),own_values(a),own_values(b)) do co,aoo,bo
    dl = zeros(eltype(d),size(d))
    co1 = zeros(eltype(d),innersize(aoo)[1],size(bo,2))
    @inbounds for i in param_eachindex(aoo)
      mul!(co1,param_getindex(aoo,i),bo)
      mul!(view(dl,:,:,i),co',co1)
    end
    dl
  end
  # Wait for the exchange to finish
  wait(t)
  # process the ghost block, accumulating onto the same per-rank buffer
  map(ld,own_values(c),own_ghost_values(a),ghost_values(b)) do dl,co,aoh,bh
    co1 = zeros(eltype(d),innersize(aoh)[1],size(bh,2))
    @inbounds for i in param_eachindex(aoh)
      mul!(co1,param_getindex(aoh,i),bh)
      mul!(view(dl,:,:,i),co',co1,1,1)
    end
  end
  copyto!(d,sreduce(ld))
  d
end

function _subfill!(a::AbstractVector,b::AbstractVector,ia,ib)
  a[ia] = b[ib]
end

function _subfill!(a::AbstractMatrix,b::AbstractMatrix,ia,ib)
  @check size(a,2) == size(b,2)
  @inbounds for k in axes(a,2)
    a[ia,k] = b[ib,k]
  end
end

function _fill_parts!(Ip,aI,a,I,l)
  _fill_index_parts!(Ip,I,l)
  _update_matrix!(aI,a,I,l)
end

function _fill_index_parts!(Ip::AbstractArray{<:LocalDofs},I,l)
  gl = I[l]
  map(Ip) do a
    if global_to_local(a.index_parts)[gl] > 0
      push!(a.global_rows,gl)
      push!(a.global_cols,l)
    end
  end
end

function _update_matrix!(aI,a,I,l)
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
      logS = (logdet_plus - sum(log,colnorms2_plus)) / (2*l)
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

function RBTransient._reduce_vector(u::PVector{<:ConsecutiveParamVector},hr_ids::AbstractVector)
  vector_partition = map(partition(u)) do lu
    RBTransient._reduce_vector(lu,hr_ids)
  end
  PVector(vector_partition,row_partition(u))
end

function RBTransient._reduce_vector(u::BlockPArray,hr_ids::AbstractVector)
  mortar(map(b -> RBTransient._reduce_vector(b,hr_ids),blocks(u)))
end

function RBTransient._reduce_trial(f::DistributedSingleFieldFESpace,hr_ids::AbstractVector)
  spaces = map(f.spaces) do s
    RBTransient._reduce_trial(s,hr_ids)
  end
  DistributedSingleFieldFESpace(spaces,f.gids,f.trian,f.vector_type,f.metadata)
end

function RBTransient._reduce_trial(f::DistributedMultiFieldFESpace,hr_ids::AbstractVector)
  field_fe_space = map(f -> RBTransient._reduce_trial(f,hr_ids),f.field_fe_space)
  part_fe_spaces = map(f -> RBTransient._reduce_trial(f,hr_ids),local_views(f.part_fe_spaces))
  DistributedMultiFieldFESpace(field_fe_space,part_fe_spaces,f.gids,f.vector_type)
end

function RBSteady._union(a::T,b::T) where T<:AbstractArray{<:AbstractVector}
  map(local_views(a),local_views(b)) do a,b
    RBSteady._union(a,b)
  end
end

function RBTransient.get_at_kron_domain(
  s::DistributedTransientSparseSnapshots,
  rowscols::Tuple,
  indices_time::AbstractVector{<:Integer}
  )

  rows,cols = rowscols
  ns = reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows))
  nt = length(indices_time)
  np = num_params(s)
  datav = map(local_values(s),local_views(rows),local_views(cols)) do s,rows,cols
    @check rows.global_cols == cols.global_cols
    data = flatten(s)
    x = zeros(eltype(data),ns*nt,np)
    if !isempty(rows)
      dof_map = get_dof_map(s)
      lrows = _remap(rows,global_to_local(rows.index_parts))
      lcols = _remap(cols,global_to_local(cols.index_parts))
      rc = sparsify_split_indices(lrows,lcols,dof_map)
      for (j,itime) in enumerate(indices_time)
        for (nzi,i) in zip(rc,rows.global_cols)
          for k in axes(data,2)
            x[(j-1)*ns+i,k] = data[nzi,k,itime]
          end
        end
      end
    end
    x
  end |> nzreduce
  ConsecutiveParamArray(datav)
end

function RBTransient.get_at_seq_domain(
  s::DistributedTransientSparseSnapshots,
  rowscols::Tuple,
  indices_time::AbstractVector{<:Integer}
  )

  rows,cols = rowscols
  n = length(indices_time)
  np = num_params(s)
  @check reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows)) == n
  datav = map(local_values(s),local_views(rows),local_views(cols)) do s,rows,cols
    @check rows.global_cols == cols.global_cols
    data = flatten(s)
    x = zeros(eltype(data),n,np)
    if !isempty(rows)
      dof_map = get_dof_map(s)
      lrows = _remap(rows,global_to_local(rows.index_parts))
      lcols = _remap(cols,global_to_local(cols.index_parts))
      rc = sparsify_split_indices(lrows,lcols,dof_map)
      for (i,(nzi,itime)) in enumerate(zip(rc,indices_time))
        for k in axes(data,2)
          x[i,k] = data[nzi,k,itime]
        end
      end
    end
    x
  end |> nzreduce
  ConsecutiveParamArray(datav)
end