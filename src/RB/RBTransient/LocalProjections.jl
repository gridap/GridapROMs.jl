function RBSteady.local_values(a::KroneckerProjection)
  map(KroneckerProjection,local_values(a.projection_space),local_values(a.projection_time))
end

function RBSteady.local_values(a::SequentialProjection)
  map(SequentialProjection,local_values(a.projection))
end

RBSteady.get_clusters(a::KroneckerProjection) = get_clusters(a.projection_space)
RBSteady.get_clusters(a::SequentialProjection) = get_clusters(a.projection)

function RBSteady.get_local(a,r::TransientRealization)
  map(r) do (μ,t)
    get_local(a,μ)
  end
end

function RBSteady.get_local(a::KroneckerProjection,μ::AbstractVector)
  KroneckerProjection(get_local(a.projection_space,μ),get_local(a.projection_time,μ))
end

function RBSteady.get_local(a::SequentialProjection,μ::AbstractVector)
  SequentialProjection(get_local(a.projection,μ))
end

function RBSteady._cluster_snaps(s::TransientSnapshotsWithIC,inds)
  initial_data = view(s.initial_data,:,inds)
  snaps = RBSteady._cluster_snaps(s.snaps,inds)
  TransientSnapshotsWithIC(initial_data,snaps)
end
