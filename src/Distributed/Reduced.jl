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
  I[1] = findrow(res)
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
    I[l] = findrow(res)
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
  I[1] = findrow(res)
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

for T in (:GenericPMatrix,:DistributedSnapshots)
  @eval begin
    function Utils.induced_norm(a::$T)
      _norm_part(x) = induced_norm(x)^2
      n = reduce(+,map(_norm_part,own_values(a)))
      sqrt(n)
    end
  end
end

for T in (:DEIMHyperReduction,:SOPTHyperReduction,:HighDimDEIMHyperReduction,:HighDimSOPTHyperReduction)
  for (A,B) in zip((:PVector,:PSparseMatrix),(:HRVecProjection,:HRMatProjection))
    @eval begin
      function RBSteady.check_interpolation(res::$A,a::$B{<:$T},fecache::AbstractArray{<:AbstractArray})
        map(local_views(res),local_views(a),local_views(fecache)) do res,a,fecache
          check_interpolation(res,a,fecache)
        end
      end
    end
  end
end

const HRPROJECTION_LABEL = "hrbasis"
const NORM_MATRIX_LABEL = "normX"
const BLOCK_LABEL = "block"
const TRIAN_LABEL = "trian"

_plabel(name,label...) = foldl(RBSteady._get_label,label;init=name)
_part_name(name,label,part) = _plabel(name,label,"part$part")
_part_filename(dir,name,label,part) = joinpath(dir,_part_name(name,label,part)*".jld")

_part_ranks(x::GenericPArray) = map(part_id,partition(axes(x,1)))
_part_ranks(x::PVector) = map(part_id,partition(axes(x,1)))

# --- GenericPArray (row-partitioned matrix / vector) ---

function _psave_pgeneric(dir,name,x::GenericPArray;label="")
  map(partition(x),partition(axes(x,1))) do xloc,ind
    serialize(_part_filename(dir,name,label,part_id(ind)),(xloc,ind))
  end
  nothing
end

function _pload_pgeneric(dir,name,ranks;label="")
  data,inds = map(ranks) do p
    deserialize(_part_filename(dir,name,label,p))
  end |> tuple_of_arrays
  GenericPArray(data,inds)
end

# --- PVector ---

function _psave_pvector(dir,name,x::PVector;label="")
  map(partition(x),partition(axes(x,1))) do xloc,ind
    serialize(_part_filename(dir,name,label,part_id(ind)),(xloc,ind))
  end
  nothing
end

function _pload_pvector(dir,name,ranks;label="")
  data,inds = map(ranks) do p
    deserialize(_part_filename(dir,name,label,p))
  end |> tuple_of_arrays
  PVector(data,inds)
end

# --- PSparseMatrix ---

function _psave_psparse(dir,name,x::PSparseMatrix;label="")
  map(partition(x),partition(axes(x,1)),partition(axes(x,2))) do xloc,rind,cind
    serialize(_part_filename(dir,name,label,part_id(rind)),(xloc,rind,cind))
  end
  nothing
end

function _pload_psparse(dir,name,ranks;label="")
  data,rinds,cinds = map(ranks) do p
    deserialize(_part_filename(dir,name,label,p))
  end |> tuple_of_arrays
  PSparseMatrix(data,rinds,cinds)
end

_haspart(dir,name,ranks;label="") = isfile(_part_filename(dir,name,label,getany(ranks)))

# --- distributed snapshots ---

function DrWatson.save(dir,s::DistributedSnapshots;label="")
  _psave_pgeneric(dir,RBSteady.SNAPSHOTS_LABEL,s.snaps;label)
end

function DrWatson.save(dir,s::DistributedBlockSnapshots;label="")
  for i in eachindex(blocks(s))
    _psave_pgeneric(dir,RBSteady.SNAPSHOTS_LABEL,blocks(s)[i].snaps;
      label=_plabel(label,"$(BLOCK_LABEL)$i"))
  end
  nothing
end

function RBSteady.load_snapshots(dir,ranks::AbstractArray;label="")
  if _haspart(dir,RBSteady.SNAPSHOTS_LABEL,ranks;label=_plabel(label,"$(BLOCK_LABEL)1"))
    _load_distributed_block_snapshots(dir,ranks;label)
  else
    DistributedSnapshots(_pload_pgeneric(dir,RBSteady.SNAPSHOTS_LABEL,ranks;label))
  end
end

function _load_distributed_block_snapshots(dir,ranks;label="")
  arr = DistributedSnapshots[]
  i = 1
  while _haspart(dir,RBSteady.SNAPSHOTS_LABEL,ranks;label=_plabel(label,"$(BLOCK_LABEL)$i"))
    snaps = _pload_pgeneric(dir,RBSteady.SNAPSHOTS_LABEL,ranks;label=_plabel(label,"$(BLOCK_LABEL)$i"))
    push!(arr,DistributedSnapshots(snaps))
    i += 1
  end
  param_data = mortar(map(get_param_data,arr))
  DistributedBlockSnapshots(arr,param_data)
end

# --- distributed projections (plain / normed / block) ---

function DrWatson.save(dir,a::DistributedPODProjection;label="")
  _psave_pgeneric(dir,RBSteady.PROJECTION_LABEL,a.basis;label)
end

function DrWatson.save(dir,a::DistributedNormedProjection;label="")
  save(dir,a.projection;label)
  _psave_psparse(dir,NORM_MATRIX_LABEL,a.norm_matrix;label)
end

function DrWatson.save(dir,a::BlockProjection{<:DistributedProjection};label="")
  for i in eachindex(a)
    save(dir,a[i];label=_plabel(label,"$(BLOCK_LABEL)$i"))
  end
  nothing
end

function RBSteady.load_projection(dir,ranks::AbstractArray;label="")
  basis = _pload_pgeneric(dir,RBSteady.PROJECTION_LABEL,ranks;label)
  proj = DistributedPODProjection(basis)
  if _haspart(dir,NORM_MATRIX_LABEL,ranks;label)
    X = _pload_psparse(dir,NORM_MATRIX_LABEL,ranks;label)
    return DistributedNormedProjection(proj,X)
  end
  return proj
end

function _load_distributed_block_projection(dir,ranks,nfields;label="")
  block_basis = map(1:nfields) do i
    RBSteady.load_projection(dir,ranks;label=_plabel(label,"$(BLOCK_LABEL)$i"))
  end
  RBSteady.BlockProjection(collect(block_basis))
end

# --- distributed hyper-reduced projections (single / block) ---

function DrWatson.save(dir,a::DistributedHRProjection;label="")
  interps = a.interpolation.interps
  map(interps,linear_indices(interps)) do interp,p
    serialize(_part_filename(dir,HRPROJECTION_LABEL,label,p),(a.basis,a.style,interp))
  end
  nothing
end

function DrWatson.save(dir,a::BlockHRProjection;label="")
  for i in eachindex(a)
    save(dir,a.array[i];label=_plabel(label,"$(BLOCK_LABEL)$i"))
  end
  nothing
end

function _load_distributed_hrprojection(dir,ranks;label="")
  loaded = map(ranks) do p
    deserialize(_part_filename(dir,HRPROJECTION_LABEL,label,p))
  end
  basis = getany(map(x -> x[1],loaded))
  style = getany(map(x -> x[2],loaded))
  interps = map(x -> x[3],loaded)
  DistributedHRProjection(basis,style,DistributedInterpolation(interps))
end

function _load_distributed_hr(dir,ranks;label="")
  if _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,"$(BLOCK_LABEL)1"))
    arr = HRProjection[]
    i = 1
    while _haspart(dir,HRPROJECTION_LABEL,ranks;label=_plabel(label,"$(BLOCK_LABEL)$i"))
      push!(arr,_load_distributed_hrprojection(dir,ranks;label=_plabel(label,"$(BLOCK_LABEL)$i")))
      i += 1
    end
    return RBSteady.BlockHRProjection(collect(arr))
  end
  _load_distributed_hrprojection(dir,ranks;label)
end

# --- distributed RB spaces ---

function RBSteady.load_reduced_subspace(dir,f::DistributedSingleFieldFESpace,ranks::AbstractArray;label="")
  basis = RBSteady.load_projection(dir,ranks;label)
  RBSteady.reduced_subspace(f,basis)
end

function RBSteady.load_reduced_subspace(dir,f::DistributedMultiFieldFESpace,ranks::AbstractArray;label="")
  basis = _load_distributed_block_projection(dir,ranks,num_fields(f);label)
  RBSteady.reduced_subspace(f,basis)
end

# --- distributed reduced contributions (rhs / lhs) ---

function DrWatson.save(dir,contrib::Contribution{V,T};label="") where {V,T<:DistributedTriangulation}
  for (i,v) in enumerate(get_contributions(contrib))
    save(dir,v;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
  nothing
end

function RBSteady.load_contribution(dir,trian::Tuple{Vararg{DistributedTriangulation}},ranks::AbstractArray;label="")
  vals = ntuple(length(trian)) do i
    _load_distributed_hr(dir,ranks;label=_plabel(label,"$(TRIAN_LABEL)$i"))
  end
  RBSteady._setup_contribution(vals,trian)
end

# --- whole distributed reduced operator ---

function RBSteady.load_operator(dir,feop::ParamOperator,ranks::AbstractArray;label="")
  test = RBSteady.load_reduced_subspace(dir,get_test(feop),ranks;label=_plabel(label,RBSteady.TEST_LABEL))
  trial = RBSteady.load_reduced_subspace(dir,get_trial(feop),ranks;label=_plabel(label,RBSteady.TRIAL_LABEL))
  trian_res = get_domains_res(feop)
  trian_jac = get_domains_jac(feop)
  red_rhs = load_contribution(dir,trian_res,ranks;label=_plabel(label,RBSteady.RHS_LABEL))
  red_lhs = load_contribution(dir,trian_jac,ranks;label=_plabel(label,RBSteady.LHS_LABEL))
  RBSteady.ReducedOperator(feop,trial,test,red_lhs,red_rhs)
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

