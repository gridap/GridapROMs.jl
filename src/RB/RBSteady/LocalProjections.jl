struct LocalProjection{A,N} <: Projection
  projections::Array{A,N}
  k::NTuple{N,KmeansResult}
end

LocalProjection(projections::AbstractVector,k::KmeansResult) = LocalProjection(projections,(k,))

const VecLocalProjection{A} = LocalProjection{A,1}
const MatLocalProjection{A} = LocalProjection{A,2}

function projection(lred::LocalReduction,s::Snapshots)
  red = get_reduction(lred)
  k = compute_clusters(lred,s)
  svec = cluster(s,k)
  proj = map(s -> projection(red,s),svec)
  LocalProjection(proj,k)
end

function projection(lred::LocalReduction,s::Snapshots,X::MatrixOrTensor)
  red = get_reduction(lred)
  k = compute_clusters(lred,s)
  svec = cluster(s,k)
  proj = map(s -> projection(red,s,X),svec)
  LocalProjection(proj,k)
end

function galerkin_projection(a::LocalProjection,b::LocalProjection)
  b̂ = map(galerkin_projection,a.projections,b.projections)
  LocalProjection(b̂,b.k)
end

function galerkin_projection(a::LocalProjection,b::LocalProjection,c::LocalProjection,args...)
  b̂ = map((pa,pb,pc) -> galerkin_projection(pa,pb,pc,args...),a.projections,b.projections,c.projections)
  LocalProjection(b̂,b.k)
end

CellData.get_domains(a::LocalProjection) = map(get_domains,a.projections)

function Utils.change_domains(a::LocalProjection,trians)
  projections′ = map(change_domains,a.projections,trians)
  LocalProjection(projections′,a.k)
end

local_values(a) = @abstractmethod
local_values(a::LocalProjection) = a.projections

function local_values(a::BlockProjection)
  litems = map(local_values,a.array)
  nlitems = length(first(litems))
  map(1:nlitems) do i
    BlockProjection(getindex.(litems,i),a.touched)
  end
end

function local_values(a::RBSpace)
  space = get_fe_space(a)
  lsubspace = local_values(get_reduced_subspace(a))
  map(x -> reduced_subspace(space,x),lsubspace)
end

local_proj_to_proj(a::Projection,b::AbstractVector{<:Projection}) = @abstractmethod
local_proj_to_proj(a::LocalProjection,b::AbstractVector{<:Projection}) = LocalProjection(b,a.k)

get_clusters(a) = @abstractmethod
get_clusters(a::LocalProjection) = a.k
get_clusters(a::BlockProjection) = get_clusters(testitem(a))
get_clusters(a::RBSpace) = get_clusters(get_reduced_subspace(a))

function get_local(a,r::Realization)
  map(r) do μ
    get_local(a,μ)
  end
end

function get_local(a::VecLocalProjection,μ::AbstractVector)
  k, = get_clusters(a)
  lab = get_label(k,μ)
  a.projections[lab]
end

function get_local(a::MatLocalProjection,μ::AbstractVector)
  k,l = get_clusters(a)
  labk = get_label(k,μ)
  labl = get_label(l,μ)
  a.projections[labk,labl]
end

function get_local(a::BlockProjection,μ::AbstractVector)
  BlockProjection(map(p -> get_local(p,μ),a.array),a.touched)
end

function get_local(a::RBSpace,μ::AbstractVector)
  space = get_fe_space(a)
  lsubspace = get_local(get_reduced_subspace(a),μ)
  reduced_subspace(space,lsubspace)
end

get_local(a::ParamOperator,μ::AbstractVector) = a
get_local(a::UncommonParamOperator,μ::AbstractVector) = a[μ]

function enrich!(
  red::SupremizerReduction{A,<:LocalReduction},
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix
  ) where A

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  a_primal_loc = local_values(a_primal)
  for j in eachindex(a_primal_loc)
    pj = a_primal_loc[j]
    for i = eachindex(a_dual)
      a_dual_i_loc = local_values(a_dual[i])
      dij = get_basis(a_dual_i_loc[j])
      C_primal_dual_i = supr_matrix[Block(1,i+1)]
      supr_i = H_primal \ C_primal_dual_i * dij
      pj = union_bases(pj,supr_i,H_primal)
    end
    a_primal_loc[j] = pj
  end
  a[1] = local_proj_to_proj(a_primal,a_primal_loc)
  return
end

function enrich!(
  red::SupremizerReduction{A,<:LocalReduction},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor
  ) where A

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  a_primal_loc = local_values(a_primal)
  for j in eachindex(a_primal_loc)
    pj = a_primal_loc[j]
    for i = eachindex(a_dual)
      a_dual_i_loc = local_values(a_dual[i])
      dij = get_cores(a_dual_i_loc[j])
      C_primal_dual_i = supr_matrix[Block(1,i+1)]
      supr_ij = tt_supremizer(H_primal,C_primal_dual_i,dij)
      pj = union_bases(pj,supr_ij,X_primal)
    end
    a_primal_loc[j] = pj
  end
  a[1] = local_proj_to_proj(a_primal,a_primal_loc)
  return
end

function reduced_residual(lred::LocalReduction,test::RBSpace,c::ArrayContribution)
  red = get_reduction(lred)
  kc = compute_clusters(lred,c)
  kr, = get_clusters(test)
  cc = cluster(c,kc)

  hr = Matrix{AffineContribution}(undef,length(kc.counts),length(kr.counts))
  for i in eachindex(kc.counts)
    ci = cc[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      testj = get_local(test,centerj)
      hr[i,j] = reduced_residual(red,testj,ci)
    end
  end

  LocalProjection(hr,(kc,kr))
end

function reduced_jacobian(lred::LocalReduction,trial::RBSpace,test::RBSpace,c::ArrayContribution)
  red = get_reduction(lred)
  kc = compute_clusters(lred,c)
  kr, = get_clusters(test)
  cc = cluster(c,kc)

  hr = Matrix{AffineContribution}(undef,length(kc.counts),length(kr.counts))
  for i in eachindex(kc.counts)
    ci = cc[i]
    for (j,centerj) in enumerate(eachcol(kr.centers))
      trialj = get_local(trial,centerj)
      testj = get_local(test,centerj)
      hr[i,j] = reduced_jacobian(red,trialj,testj,ci)
    end
  end

  LocalProjection(hr,(kc,kr))
end

function cluster(a,i)
  @abstractmethod
end

function cluster(a,red::LocalReduction)
  r = _get_realization(a)
  k = compute_clusters(r,red)
  cluster(a,k)
end

function cluster(a,k::KmeansResult)
  labels = get_label(k,a)
  cluster_ids = group_ilabels(labels)
  cluster_cache = array_cache(cluster_ids)
  _ids = getindex!(cluster_cache,cluster_ids,1)
  S = typeof(cluster(a,_ids))
  cache = Vector{S}(undef,length(cluster_ids))

  for icluster in 1:length(cluster_ids)
    ids = getindex!(cluster_cache,cluster_ids,icluster)
    cache[icluster] = cluster(a,ids)
  end

  return cache
end

function cluster(r::Realization,inds::AbstractVector)
  r[inds]
end

function cluster(s::ArrayContribution,inds::AbstractVector)
  contribution(get_domains(s)) do trian
    cluster(s[trian],inds)
  end
end

function cluster(s::GenericSnapshots,inds::AbstractVector)
  sinds = select_snapshots(s,inds)
  data = collect(get_all_data(sinds))
  GenericSnapshots(data,get_dof_map(sinds),get_realization(sinds))
end

function cluster(s::ReshapedSnapshots,inds::AbstractVector)
  sinds = select_snapshots(s,inds)
  data = collect(get_all_data(sinds))
  ReshapedSnapshots(data,get_param_data(sinds),get_dof_map(sinds),get_realization(sinds))
end

function cluster(s::BlockSnapshots,inds::AbstractVector)
  array = Array{Snapshots,ndims(s)}(undef,size(s))
  touched = s.touched
  for i in eachindex(touched)
    if touched[i]
      array[i] = cluster(s[i],inds)
    end
  end
  return BlockSnapshots(array,touched)
end

# cannot provide Kmeans in the following cases, as these types do not store realizations

function cluster(a::AbstractParamArray{T,N},labels::AbstractVector) where {T,N}
  S = ConsecutiveParamArray{T,N}
  cluster_labels = group_labels(labels)
  cache = Vector{S}(undef,length(cluster_labels))
  for icluster in 1:length(cluster_labels)
    pini = cluster_labels.ptrs[icluster]
    pend = cluster_labels.ptrs[icluster+1]-1
    cache[icluster] = ParamArray(a[pini:pend])
  end
  return cache
end

function cluster(a::BlockParamArray{T,N},labels::AbstractVector) where {T,N}
  S = typeof(mortar(map(a -> ParamArray(a[1:1]),blocks(a))))
  cluster_labels = group_labels(labels)
  cache = Vector{S}(undef,length(cluster_labels))
  for icluster in 1:length(cluster_labels)
    pini = cluster_labels.ptrs[icluster]
    pend = cluster_labels.ptrs[icluster+1]-1
    cache[icluster] = mortar(map(a -> ParamArray(a[pini:pend]),blocks(a)))
  end
  return cache
end

function cluster(a::RBParamVector,labels::AbstractVector)
  data = cluster(a.data,labels)
  fe_data = cluster(a.fe_data,labels)
  map(RBParamVector,data,fe_data)
end

function cluster_sort(a,labels::AbstractVector)
  cluster_ids = group_ilabels(labels)
  _cluster_sort(a,cluster_ids.data)
end

# utils

function compute_clusters(red::LocalReduction,r::AbstractRealization)
  Random.seed!(1234)
  pmat = _get_params_marix(r)
  k = kmeans(pmat,red.ncentroids)
  return k
end

function compute_clusters(red::LocalReduction,s::AbstractSnapshots)
  compute_clusters(red,get_realization(s))
end

function compute_clusters(red::LocalReduction,s::ArrayContribution)
  compute_clusters(red,first(s.values))
end

function get_label(k::KmeansResult,a)
  get_label(k,_get_realization(a))
end

function get_label(k::KmeansResult,r::Realization)
  map(r) do μ
    get_label(k,μ)
  end
end

function get_label(k::KmeansResult,x::AbstractVector{<:Number})
  dists = centroid_distances(k,x)
  argmin(dists)
end

get_centers(k::KmeansResult) = eachcol(k.centers)

function centroid_distances(k::KmeansResult,x::AbstractVector{<:Number})
  centers = get_centers(k)
  dists = zeros(length(centers))
  for (i,y) in enumerate(centers)
    dists[i] = norm(x-y)
  end
  return dists
end

function centroid_distances(k::KmeansResult,x::AbstractMatrix)
  dists = zeros(size(x,2))
  for i in axes(x,2)
    xi = view(x,:,i)
    dists[i] = centroid_distances(k,xi)
  end
  return dists
end

function compute_ncentroids(
  r::AbstractRealization;
  init=min(4,num_params(r)),
  iend=min(16,floor(Int,num_params(r)/2))
  )

  Random.seed!(1234)
  pmat = _get_params_marix(r)
  kvars = zeros(iend-init+1)
  all_ncentroids = init:iend
  for (i,ncentroids) in enumerate(all_ncentroids)
    k = kmeans(pmat,ncentroids)
    kvars[i] = kmeans_variance(k,pmat)
  end
  elbow = _compute_elbow(kvars)
  all_ncentroids[elbow]
end

for f in (:(DofMaps.group_labels),:(DofMaps.get_group_to_labels),
          :(DofMaps.group_ilabels),:(DofMaps.get_group_to_ilabels))
  @eval begin
    function $f(q,k::KmeansResult)
      labels = get_label(k,q)
      $f(labels)
    end
  end
end

function kmeans_variance(k::KmeansResult,pmat::AbstractMatrix)
  errs = 0.0
  for α in eachcol(pmat)
    lab = get_label(k,α)
    β = view(k.centers,:,lab)
    errs += norm(α-β)^2
  end
  return errs
end

function _compute_elbow(v::AbstractVector)
  dv = zeros(length(v)-1)
  for i in 1:length(dv)
    dv[i] = abs(v[i+1]-v[i]) / v[i]
  end
  argmin(dv)
end

_get_realization(r::AbstractRealization) = get_params(r)
_get_realization(s::AbstractSnapshots) = get_realization(s)
_get_realization(a::ArrayContribution) = get_realization(first(a.values))

_get_params_marix(r::Realization) = stack(r.params)
_get_params_marix(r::AbstractRealization) = _get_params_marix(get_params(r))

_cluster_sort(a,inds::AbstractVector) = @abstractmethod

function _cluster_sort(a::AbstractParamVector,inds::AbstractVector)
  data = map(i -> a[i],inds)
  ParamArray(data)
end

function _cluster_sort(a::BlockParamVector,inds::AbstractVector)
  mortar(map(a -> _cluster_sort(a,inds),blocks(a)))
end

function _cluster_sort(a::RBParamVector,inds::AbstractVector)
  data = _cluster_sort(a.data,inds)
  fe_data = _cluster_sort(a.fe_data,inds)
  RBParamVector(data,fe_data)
end
