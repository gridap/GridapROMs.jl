struct LocalProjection{A} <: Projection
  projections::Vector{A}
  k::KmeansResult
end

function Projection(lred::LocalReduction,s::AbstractArray,args...)
  red = get_reduction(lred)
  k = compute_clusters(lred,s)
  svec = cluster_snapshots(s,k)
  proj = map(s -> Projection(red,s,args...),svec)
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

function empirical_interpolation(a::LocalProjection)
  map(empirical_interpolation,a.projections) |> tuple_of_arrays
end

local_values(a) = @abstractmethod
local_values(a::LocalProjection) = a.projections
local_values(a::NormedProjection) = map(p->NormedProjection(p,a.norm_matrix),local_values(a.projection))

function local_values(a::RBSpace)
  space = get_fe_space(a)
  lsubspace = local_values(get_reduced_subspace(a))
  map(x -> reduced_subspace(space,x),lsubspace)
end

get_clusters(a) = @abstractmethod
get_clusters(a::LocalProjection) = a.k
get_clusters(a::NormedProjection) = get_clusters(a.projection)
get_clusters(a::RBSpace) = get_clusters(get_reduced_subspace(a))

function get_local(a,r::AbstractRealization)
  map(r) do μ
    get_local(a,μ)
  end
end

function get_local(a::LocalProjection,μ::AbstractVector)
  k = get_clusters(a)
  lab = get_label(k,μ)
  a.projections[lab]
end

function get_local(a::NormedProjection,μ::AbstractVector)
  NormedProjection(get_local(a.projection,μ),a.norm_matrix)
end

function get_local(a::RBSpace,μ::AbstractVector)
  space = get_fe_space(a)
  lsubspace = get_local(get_reduced_subspace(a),μ)
  reduced_subspace(space,lsubspace)
end

function reduced_residual(lred::LocalReduction,test::RBSpace,c::ArrayContribution)
  red = get_reduction(lred)
  k = get_clusters(test)
  cc = cluster_snapshots(c,k)
  hr = map(local_values(test),cc) do test,c
    reduced_residual(red,test,c)
  end
  LocalProjection(hr,k)
end

function reduced_jacobian(lred::LocalReduction,trial::RBSpace,test::RBSpace,c::ArrayContribution)
  red = get_reduction(lred)
  k = get_clusters(test)
  cc = cluster_snapshots(c,k)
  hr = map(local_values(trial),local_values(test),cc) do trial,test,c
    reduced_jacobian(red,trial,test,c)
  end
  LocalProjection(hr,k)
end

# local utils

function compute_clusters(red::LocalReduction,r::AbstractRealization)
  pmat = _get_params_marix(r)
  k = kmeans(pmat,red.ncentroids)
  return k
end

function compute_clusters(red::LocalReduction,s::Snapshots)
  compute_clusters(red,get_realization(s))
end

function cluster_snapshots(red::LocalReduction,s::ArrayContribution)
  r = get_realization(first(s.values))
  k = compute_clusters(red,r)
  cluster_snapshots(s,k)
end

function cluster_snapshots(red::LocalReduction,s::Snapshots)
  r = get_realization(s)
  k = compute_clusters(red,r)
  cluster_snapshots(s,k)
end

for T in (:ArrayContribution,:Snapshots)
  @eval begin
    function cluster_snapshots(s::$T,k::KmeansResult)
      cache = return_cache(cluster_snapshots,s,k)
      evaluate!(cache,cluster_snapshots,s,k)
      return cache
    end

    function Arrays.return_cache(::typeof(cluster_snapshots),s::$T,k::KmeansResult)
      ncenters = size(k.centers,2)
      cluster = Int[1]
      T = typeof(_cluster_snaps(s,cluster))
      Vector{T}(undef,ncenters)
    end

    function Arrays.evaluate!(cache,::typeof(cluster_snapshots),s::$T,k::KmeansResult)
      a = assignments(k)
      ncenters = size(k.centers,2)
      for label in 1:ncenters
        cluster = findall(a .== label)
        cache[label] = _cluster_snaps(s,cluster)
      end
      return cache
    end
  end
end

function get_label(k::KmeansResult,x::AbstractVector)
  dists = centroid_distances(k,x)
  argmin(dists)
end

get_centers(k::KmeansResult) = eachcol(k.centers)

function centroid_distances(k::KmeansResult,x::AbstractVector)
  centers = get_centers(k)
  dists = zeros(length(centers))
  for (i,y) in enumerate(centers)
    dists[i] = norm(x-y)
  end
  return dists
end

_get_params_marix(r::Realization) = stack(r.params)
_get_params_marix(r::AbstractRealization) = _get_params_marix(get_params(r))

function _cluster_snaps(s::ArrayContribution,inds)
  contribution(get_domains(s)) do trian
    _cluster_snaps(s[trian],inds)
  end
end

function _cluster_snaps(s::GenericSnapshots,inds)
  sinds = select_snapshots(s,inds)
  data = collect(get_all_data(sinds))
  GenericSnapshots(data,get_dof_map(sinds),get_realization(sinds))
end

function _cluster_snaps(s::ReshapedSnapshots,inds)
  sinds = select_snapshots(s,inds)
  data = collect(get_all_data(sinds))
  ReshapedSnapshots(data,get_param_data(sinds),get_dof_map(sinds),get_realization(sinds))
end
