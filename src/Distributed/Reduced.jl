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

# projections

PartitionedArrays.partition(a::Projection) = partition(get_basis(a))
PartitionedArrays.local_values(a::Projection) = local_values(get_basis(a))
PartitionedArrays.own_values(a::Projection) = own_values(get_basis(a))
PartitionedArrays.ghost_values(a::Projection) = ghost_values(get_basis(a))
PartitionedArrays.consistent!(a::Projection) = consistent!(get_basis(a))

struct DistributedPODProjection <: Projection
  basis::AbstractMatrix
end

function RBSteady.PODProjection(basis::GenericPMatrix)
  DistributedPODProjection(basis)
end

function RBSteady.Projection(basis::GenericPMatrix,s::DistributedSparseSnapshots)
  basis′ = recast(basis,s)
  DistributedPODProjection(basis′)
end

function RBTransient.kron_projection(red::KroneckerReduction,s::DistributedSparseSnapshots,args...)
  basis_space,basis_time = tucker(red.reductions,s,args...)
  basis_space′ = recast(basis_space,s)
  projection_space = PODProjection(basis_space′)
  projection_time = PODProjection(basis_time)
  return projection_space,projection_time
end

RBSteady.get_basis(a::DistributedPODProjection) = a.basis
RBSteady.fe_dof_ids(a::DistributedPODProjection) = row_partition(a)
row_partition(a::DistributedPODProjection) = row_partition(a.basis)
col_partition(a::DistributedPODProjection) = col_partition(a.basis)
flat_row_partition(a::DistributedPODProjection) = flat_row_partition(a.basis)

RBSteady.projection_type(a::DistributedPODProjection) = PVector{Vector{projection_eltype(a)}}

function Algebra.allocate_vector(::Type{<:PVector{V}},rows::AbstractVector) where V
  allocate_vector(V,rows)
end

function Algebra.allocate_in_domain(a::Projection,x::PVector{<:V}) where V<:AbstractParamVector
  x̂ = allocate_vector(PVector{eltype(V)},RBSteady.reduced_dof_ids(a))
  return parameterise(x̂,param_length(x))
end

function Algebra.allocate_in_range(a::Projection,x̂::PVector{<:V}) where V<:AbstractParamVector
  x = allocate_vector(PVector{eltype(V)},RBSteady.fe_dof_ids(a))
  return parameterise(x,param_length(x̂))
end

function RBSteady.allocate_full_matrix(::Type{<:GenericPArray{M}},rows::PRange,cols::AbstractVector) where M
  GenericPArray{M}(undef,partition(rows),cols)
end

function RBSteady.union_bases(a::DistributedPODProjection,b::DistributedPODProjection,args...) 
  union_bases(a,get_basis(b),args...)
end

function RBSteady.union_bases(a::DistributedPODProjection,basis_b::AbstractMatrix,args...)
  basis_a = get_basis(a)
  basis_ab = gram_schmidt(basis_b,basis_a,args...)
  DistributedPODProjection(basis_ab)
end

function RBSteady.galerkin_projection(a::DistributedPODProjection,b::DistributedPODProjection)
  lb̂ = map(own_values(a),own_values(b)) do ao,bo
    galerkin_projection(ao,bo)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady.galerkin_projection(a::DistributedPODProjection,b::DistributedPODProjection,c::DistributedPODProjection,args...)
  lb̂ = map(own_values(a),own_values(b),own_values(c)) do ao,bo,co
    galerkin_projection(ao,bo,co,args...)
  end
  b̂ = reduce(+,lb̂)
  return ReducedProjection(b̂)
end

function RBSteady._allocate_projection(red::Reduction,s::DistributedBlockSnapshots{N}) where N
  T = DistributedPODProjection
  block_basis = Array{T,N}(undef,size(s))
  BlockProjection(block_basis,s.touched)
end

function GridapDistributed.local_views(a::DistributedPODProjection)
  map(local_views(a.basis)) do basis
    PODProjection(basis)
  end
end

function GridapDistributed.local_views(a::NormedProjection)
  map(local_views(a.projection),local_views(a.norm_matrix)) do proj,norm
    NormedProjection(proj,norm)
  end
end

# reduced basis spaces

const DistributedRBSpace{S<:DistributedFESpace} = RBSpace{S}

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

# integration domains

struct LocalRows{T,Ti} <: AbstractVector{T}
  rows::Vector{T}
  inds::Vector{Ti}
end

LocalRows() = LocalRows(Int[],Int[])

Base.size(a::LocalRows) = size(a.rows)
Base.IndexStyle(::Type{<:LocalRows}) = IndexLinear()
Base.getindex(a::LocalRows,i::Int) = getindex(a.rows,i)
Base.setindex!(a::LocalRows,v,i::Int) = setindex!(a.rows,v,i)

function RBSteady._evaluate!(a,cellrows,rows::LocalRows)
  fill!(a,zero(eltype(a)))
  for (irow,row) in enumerate(rows)
    for (icellrow,cellrow) in enumerate(cellrows)
      if row == cellrow
        a[icellrow] = rows.inds[irow]
      end
    end
  end
  a
end

function RBSteady._evaluate!(a,cellrows,cellcols,rows::LocalRows,cols::LocalRows)
  fill!(a,zero(eltype(a)))
  ncellrows = length(cellrows)
  for (irowcol,rowcol) in enumerate(zip(rows,cols))
    row,col = rowcol
    for (icellrow,cellrow) in enumerate(cellrows)
      for (icellcol,cellcol) in enumerate(cellcols)
        if row == cellrow && col == cellcol
          icellrowcol = icellrow + (icellcol-1)*ncellrows
          a[icellrowcol] = rows.inds[irowcol]
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
  Iloc = map(_ -> LocalRows(),parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = argmax(abs,res)
  _push_to_local!(Iloc,parts,I,1)
  _from_submatrix!(basisI,basis,I,1)
  for l = 2:n
    PᵀU = view(basisI,1:l-1,1:l-1)
    Pᵀuₗ = view(basisI,1:l-1,l)
    c = vec(PᵀU \ Pᵀuₗ)
    map(own_values(res),own_values(basis)) do ro,bo
      @. ro = bo[:,l]
      mul!(ro,view(bo,:,1:l-1),c,-1.0,1.0)
    end
    I[l] = argmax(abs,res)
    _push_to_local!(Iloc,parts,I,l)
    _from_submatrix!(basisI,basis,I,l)
  end
  return Iloc,basisI
end

function RBSteady.SOPT(basis::GenericPMatrix)
  T = eltype(basis)
  n = size(basis,2)
  I = zeros(Int,n)  
  parts = partition(axes(basis,1))
  Iloc = map(_ -> LocalRows(),parts)
  basisI = zeros(T,n,n)
  res = GenericPArray{Vector{T}}(undef,parts)
  map(own_values(res),own_values(basis)) do ro,bo
    @. ro = bo[:,1]
  end
  I[1] = argmax(abs,res)
  _push_to_local!(Iloc,parts,I,1)
  _from_submatrix!(basisI,basis,I,1)
  for l in 2:n
    P = I[1:l-1]
    PᵀU = view(basisI,1:l-1,1:l)
    G = PᵀU'*PᵀU
    colnorms2 = vec(sum(abs2,PᵀU;dims=1))
    Il = _best_s_opt_index(basis,P,G,colnorms2,l)
    @check Il > 0
    I[l] = Il
    _push_to_local!(Iloc,parts,I,l)
    _from_submatrix!(basisI,basis,I,l)
  end
  return Iloc,basisI
end

for f in (:DEIM,:SOPT)
  @eval begin
    function RBSteady.$f(A::PSparseMatrix)
      B = get_all_data(A)
      I,AI = $f(B)
      R′,C′ = map(local_views(I),local_values(A),flat_row_partition(B)) do I,A,rci
        _remap!(I,global_to_local(rci))
        R′,C′ = recast_split_indices(I,testitem(A))
        _remap!(R′,local_to_global(row_partition(rci)))
        _remap!(C′,local_to_global(col_partition(rci)))
        R′,C′
      end |> tuple_of_arrays
      return (R′,C′),AI
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
    _remap!(rows,global_to_local(gids))
    IntegrationDomain(trian,test,rows)
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
    _remap!(rows,global_to_local(rgids))
    _remap!(cols,global_to_local(cgids))
    IntegrationDomain(trian,trial,test,rows,cols)
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
    @check rows.inds == cols.inds
    if !isempty(rows)
      sparsity = get_sparsity(get_dof_map(s))
      rc = sparsify_split_indices(rows,cols,sparsity)
      LocalRows(rc,rows.inds)
    else
      LocalRows()
    end
  end
  get_at_domain(s,inds)
end

function RBSteady.get_at_domain(a::GenericPArray,rows::AbstractArray{<:LocalRows})
  n = size(a,2)
  @check reduce(+,map(length,rows)) == n
  datav = zeros(eltype(a),n,n)
  map(local_values(a),row_partition(a),local_views(rows)) do data,rparts,rows
    _remap!(rows,global_to_local(rparts))
    if !isempty(rows.rows)
      for (vi,i) in zip(rows.rows,rows.inds)
        for k in axes(data,2)
          datav[vi,k] = data[i,k]
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
  coeff = reduce(+,_coeff)
  mul!(b̂,a,coeff,o,o)
  return b̂
end

function FESpaces.interpolate!(
  b̂::AbstractArray,
  _coeff::AbstractArray{<:AbstractArray},
  a::DistributedHRProjection{A,NoHyperReduction} where A,
  x::AbstractArray{<:AbstractArray}
  )

  coeff = reduce(+,_coeff)
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

# post process 

function DrWatson.save(dir,s::DistributedSnapshots;label="")
  map(local_values(s),row_partition(s)) do s,row
    part = part_id(row)
    save(dir,s;label=_get_label(label,"part",part))
  end
end

function DrWatson.save(dir,a::DistributedPODProjection;label="")
  map(local_values(a),row_partition(a)) do a,row
    part = part_id(row)
    save(dir,a;label=_get_label(label,"part",part))
  end
end

function DrWatson.save(dir,contrib::Contribution{V};label="") where V<:DistributedHRProjection
  map(local_values.(contrib)) do a,row
    part = part_id(row)
    save(dir,a;label=_get_label(label,"part",part))
  end
end

for T in (:DistributedSteadySnapshots,:DistributedTransientSnapshots)
  @eval begin
    function Utils.compute_relative_error(sol::$T,sol_approx::$T)
      errs = map(own_values(sol),own_values(sol_approx)) do sol,sol_approx
        compute_relative_error(sol,sol_approx)^2
      end
      sqrt(reduce(+,errs))
    end

    function Utils.compute_relative_error(sol::$T,sol_approx::$T,X::PSparseMatrix)
      errs = map(own_values(sol),own_values(sol_approx),own_values(X)) do sol,sol_approx,X
        compute_relative_error(sol,sol_approx,X)^2
      end
      sqrt(reduce(+,errs))
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
  map(own_values(a),partition(axes(a,1))) do oa,ra
    g2o = global_to_own(ra)
    for k in l 
      or = g2o[I[k]]
      or > 0 && _subfill!(aI,oa,k,or)
    end
  end
end

function _push_to_local!(Iloc::AbstractArray{<:LocalRows},row_parts,I,l)
  gl = I[l]
  map(Iloc,row_parts) do li,ri
    ol = global_to_own(ri)[gl]
    if ol > 0
      push!(li.rows,gl)
      push!(li.inds,l)
    end
  end
end

function _remap!(x,x_to_y)
  for (i,xi) in enumerate(x)
    x[i] = x_to_y[xi]
  end
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