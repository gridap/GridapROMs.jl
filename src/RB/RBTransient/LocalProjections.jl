function RBSteady.local_values(a::KroneckerProjection)
  map(KroneckerProjection,local_values(a.projection_space),local_values(a.projection_time))
end

function RBSteady.local_values(a::SequentialProjection)
  map(SequentialProjection,local_values(a.projection))
end

RBSteady.get_clusters(a::KroneckerProjection) = get_clusters(a.projection_space)
RBSteady.get_clusters(a::SequentialProjection) = get_clusters(a.projection)

RBSteady.get_local(a,r::TransientRealization) = get_local(a,get_params(r),get_times(t))
RBSteady.get_local(a,μt::Tuple{Any,Any}) = get_local(a,first(μt))

function RBSteady.get_local(a::KroneckerProjection,μ::AbstractVector)
  KroneckerProjection(get_local(a.projection_space,μ),get_local(a.projection_time,μ))
end

function RBSteady.get_local(a::SequentialProjection,μ::AbstractVector)
  SequentialProjection(get_local(a.projection,μ))
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,<:LocalReduction{<:KroneckerReduction}},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor
  ) where A

  @check a.touched[1] "Primal field not defined"
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = cholesky(X_primal)
  a_primal_space_loc,a_primal_time_loc = local_values(a_primal)
  for j in eachindex(a_primal_space_loc)
    pj_space = a_primal_space_loc[j]
    pj_time = a_primal_time_loc[j]
    for i = eachindex(a_dual)
      if a.touched[i]
        a_dual_i_space_loc,a_dual_i_time_loc = local_values(a_dual[i])
        dij_space = get_basis(a_dual_i_space_loc[j])
        C_primal_dual_i = supr_matrix[Block(1,i+1)]
        supr_space_i = H_primal \ C_primal_dual_i * dij_space
        pj_space = union_bases(pj_space,supr_space_i,H_primal)

        dij_time = get_basis_time(a_dual_i_time_loc[j])
        pj_time = time_enrichment(red,pj_time,dij_time;kwargs...)
      end
    end
    a_primal_space_loc[j] = pj_space
    a_primal_time_loc[j] = pj_time
  end
  return
end

function RBSteady._cluster_snaps(s::TransientSnapshotsWithIC,inds)
  initial_data = view(s.initial_data,:,inds)
  snaps = RBSteady._cluster_snaps(s.snaps,inds)
  TransientSnapshotsWithIC(initial_data,snaps)
end
