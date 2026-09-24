# reduced basis spaces

const DistributedRBSpace{S<:DistributedFESpace} = RBSpace{S}

const DistributedSingleFieldRBSpace{S<:DistributedSingleFieldFESpace} = DistributedRBSpace{S}
const DistributedMultiFieldRBSpace{S<:DistributedMultiFieldFESpace} = DistributedRBSpace{S}

function GridapDistributed.local_views(r::DistributedSingleFieldRBSpace)
  map(local_views(r.space),local_views(r.subspace)) do space,subspace
    RBSpace(space,subspace)
  end
end

# `r.subspace` is a `BlockProjection` here, whose fields may have different
# concrete types across blocks (e.g. one field null, one not); `local_views`
# is deliberately not defined for `BlockProjection` itself (it is used in
# both distributed and non-distributed contexts), so the per-field split and
# per-rank recombination happens here instead, scoped to `DistributedRBSpace`
function GridapDistributed.local_views(r::DistributedMultiFieldRBSpace)
  subspace = get_reduced_subspace(r)
  local_subspaces = map(local_views,subspace.array)
  map(local_views(r.space),local_subspaces...) do space,fields...
    RBSpace(space,BlockProjection(collect(fields)))
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

# the generic `inv_project(a::Projection,x̂) = allocate_in_range(a,x̂); ...` allocates
# each block's output `PVector` independently from that field's *stored basis*
# (`RBSteady.fe_dof_ids(a) = axes(get_basis(a),1)`), which is not guaranteed to carry
# the same `PRange` element type across fields. `GridapDistributed.BlockPRange` can't
# represent that regardless (it requires `Vector{<:PRange{A}}` for a single `A`), yet
# this is exactly the situation for a `BlockMultiFieldStyle` space here. Sidestep it
# entirely by reusing the FE space's own, already-homogeneous `BlockPRange` (built by
# `GridapDistributed` itself via `generate_multi_field_gids`, stored as `fs.gids`) as
# the allocation template, mirroring `GridapDistributed.Algebra.allocate_vector(
# ::Type{<:BlockPVector{V}},ids::BlockPRange) = BlockPVector{V}(undef,ids)`, which
# builds each block directly off `ids` without ever re-`mortar`-ing the row partition
function RBSteady.inv_project(r::DistributedMultiFieldRBSpace,x̂::AbstractVector)
  a = get_reduced_subspace(r)
  gids = get_free_dof_ids(get_fe_space(r))
  x = allocate_vector(BlockPVector{eltype(x̂)},gids)
  x = parameterise(x,param_length(x̂))
  inv_project!(x,a,x̂)
  return x
end

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

struct DistributedIntegrationDomain{A} <: Interpolation
  domains::A
end

GridapDistributed.local_views(a::DistributedIntegrationDomain) = a.domains

function RBSteady.IntegrationDomain(
  trian::DistributedTriangulation,
  test::DistributedRBSpace,
  rows::AbstractArray{<:AbstractVector}
  )

  rgids = get_free_dof_ids(test)
  domains = map(
    local_views(trian),
    local_views(test),
    local_views(rows),
    local_views(rgids),
    ) do trian,test,rows,rgids
    lrows = _remap(rows,global_to_local(rgids))
    domain = IntegrationDomain(trian,test,lrows)
    grows = _remap(lrows,local_to_global(rgids))
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

# a structurally-null (e.g. identically-zero) projection is not wrapped by the
# generic `Interpolation(red,a::Projection,args...)` fallback into a
# `DistributedInterpolation`, since it short-circuits before ever touching
# `trian`/`test`. It still needs to be wrapped here: a plain `EmptyInterpolation`
# and a `DistributedInterpolation` mix badly in `get_integration_cells(a::
# BlockInterpolation)`'s per-block `_union` (a real block's cells are
# `MPIArray`-per-rank; a bare `EmptyInterpolation`'s are a plain `Vector`, and
# `_union` requires matching types) whenever a multi-field block mixes a null
# sub-projection (e.g. Stokes' pressure-pressure block) with real ones. Since
# the null-ness of a distributed projection is consistent across ranks,
# replicate the (empty) interpolation on every rank explicitly; the resulting
# `DistributedInterpolation`-of-`EmptyInterpolation`s is itself trivial to spot
# downstream (see `check_interpolation` in `Distributed/PostProcess.jl`)
for T in (:DEIMHyperReduction,:SOPTHyperReduction)
  @eval begin
    function RBSteady.Interpolation(
      red::$T,
      a::Projection,
      trian::DistributedTriangulation,
      test::DistributedRBSpace
      )

      if isnull(a)
        interps = map(local_views(trian)) do _
          EmptyInterpolation()
        end
        return DistributedInterpolation(interps)
      end
      GreedyInterpolation(red,a,trian,test)
    end

    function RBSteady.Interpolation(
      red::$T,
      a::Projection,
      trian::DistributedTriangulation,
      trial::DistributedRBSpace,
      test::DistributedRBSpace
      )

      if isnull(a)
        interps = map(local_views(trian)) do _
          EmptyInterpolation()
        end
        return DistributedInterpolation(interps)
      end
      GreedyInterpolation(red,a,trian,trial,test)
    end
  end
end

function RBSteady.GreedyInterpolation(interp,domain::DistributedIntegrationDomain)
  interps = map(local_views(domain)) do di
    GreedyInterpolation(interp,di)
  end
  DistributedInterpolation(interps)
end

GridapDistributed.local_views(a::DistributedInterpolation) = a.interps

for f in (:get_integration_cells,:get_cell_idofs)
  @eval begin
    function RBSteady.$f(a::DistributedInterpolation)
      map(local_views(a)) do a
        $f(a)
      end
    end
  end
end

function RBSteady.get_owned_integration_cells(a::DistributedInterpolation,args...)
  map(local_views(a)) do a
    get_owned_integration_cells(a,args...)
  end
end

function RBSteady.get_interpolation_dofs(a::DistributedInterpolation)
  _unpack(x) = x
  _unpack(x::AbstractArray{<:Tuple}) = tuple_of_arrays(x) 
  dofs = map(local_views(a)) do a
    get_interpolation_dofs(a)
  end
  _unpack(dofs)
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

const TransientDistributedInterpolation{A} = TransientInterpolation{A,<:DistributedInterpolation}

function GridapDistributed.local_views(a::TransientDistributedInterpolation)
  map(local_views(a.interp_space)) do interp
    TransientInterpolation(a.style,interp,a.indices_time)
  end
end

function FESpaces.interpolate!(
  cache::AbstractArray{<:AbstractArray},
  a::TransientDistributedInterpolation,
  b::AbstractArray{<:AbstractArray}
  )

  map(local_views(cache),local_views(a),local_views(b)) do cache,interp,b
    interpolate!(cache,interp,b)
  end
end

function RBSteady.get_at_domain(s::DistributedSnapshots,rows::AbstractVector{<:LocalDofs})
  n = reduce(max,map(r -> isempty(r.global_cols) ? 0 : maximum(r.global_cols),rows))
  np = num_params(s)
  datav = map(local_values(s),local_views(rows)) do s,rows
    data = flatten(s)
    x = zeros(eltype(data),n,np)
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

const DistributedHRProjection{
  A<:HyperReduction,
  B<:Projection,
  C<:Union{DistributedInterpolation,TransientDistributedInterpolation}
} = RBSteady.GenericHRProjection{A,B,C}

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

function FESpaces.interpolate!(
  hypred::AbstractArray,
  coeff::AbstractArray,
  a::RBSteady.BlockHRProjection,
  b::BlockPArray
  )

  for i in eachindex(a)
    interpolate!(blocks(hypred)[i],coeff[i],a.array[i],blocks(b)[i])
  end
  return hypred
end

function RBSteady.collect_cell_hr_matrix(
  trial::DistributedRBSpace,
  test::DistributedRBSpace,
  a::DistributedDomainContribution,
  strian::DistributedTriangulation,
  interp::Interpolation
  )

  cell_idofs = get_cell_idofs(interp)
  cells = get_owned_integration_cells(interp,strian)
  cell_mat_rc = map(local_views(trial),local_views(test),local_views(a),local_views(strian)) do trial,test,a,strian
    scell_mat = get_contribution(a,strian)
    cell_mat,trian = move_contributions(scell_mat,strian)
    @assert ndims(eltype(cell_mat)) == 2
    cell_mat_c = attach_constraints_cols(trial,cell_mat,trian)
    attach_constraints_rows(test,cell_mat_c,trian)
  end
  (cell_mat_rc,cell_idofs,cells)
end

function RBSteady.collect_cell_hr_vector(
  test::DistributedRBSpace,
  a::DistributedDomainContribution,
  strian::DistributedTriangulation,
  interp::Interpolation
  )

  cell_idofs = get_cell_idofs(interp)
  cells = get_owned_integration_cells(interp,strian)
  cell_vec_r = map(local_views(test),local_views(a),local_views(strian)) do test,a,strian
    scell_vec = get_contribution(a,strian)
    cell_vec,trian = move_contributions(scell_vec,strian)
    @assert ndims(eltype(cell_vec)) == 1
    attach_constraints_rows(test,cell_vec,trian)
  end
  (cell_vec_r,cell_idofs,cells)
end

for T in (:MPIArray,:DebugArray)
  @eval begin
    function RBSteady.assemble_hr_array_add!(A::$T,cellvals::$T,celldofs::$T,cells::$T,args...)
      map(A,cellvals,celldofs,cells) do A,cellvals,celldofs,cells
        assemble_hr_array_add!(A,cellvals,celldofs,cells,args...)
      end
    end

    function RBSteady.fetch_block(a::$T,i::Int)
      map(local_views(a)) do a
        fetch_block(a,i)
      end
    end
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

function Utils.ChildTriangulation(t::DistributedTriangulation,inds::AbstractArray{<:AbstractVector})
  models = get_background_model(t)
  trians = map(local_views(t),local_views(inds)) do t,inds
    ChildTriangulation(t,inds)
  end
  DistributedTriangulation(trians,models;metadata=t.metadata)
end

function Utils.ChildTriangulation(t::DistributedTriangulation,inds::Vector)
  models = get_background_model(t)
  trians = map(local_views(t)) do t
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

function RBSteady.get_integration_cells(t::DistributedTriangulation)
  map(local_views(t)) do t
    get_integration_cells(t)
  end
end

# needed 

function Base.fill!(a::DebugArray,b::Number)
  map(local_views(a)) do a
    fill!(a,b)
  end
  a
end

function Base.fill!(a::MPIArray,b::Number)
  map(local_views(a)) do a
    fill!(a,b)
  end
  a
end

function Base.fill!(a::Array{<:MPIArray},b::Number)
  for ai in a
    fill!(ai,b)
  end
  a
end

function Base.fill!(a::Array{<:DebugArray},b::Number)
  for ai in a
    fill!(ai,b)
  end
  a
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