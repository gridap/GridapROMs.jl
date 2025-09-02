function RBSteady.local_vals(a::KroneckerProjection)
  map(KroneckerProjection,local_vals(a.projection_space),local_vals(a.projection_time))
end

function RBSteady.local_vals(a::SequentialProjection)
  map(SequentialProjection,local_vals(a.projection))
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

function RBSteady.get_local(a::Tuple{Vararg{Any}},μ::AbstractVector)
  map(a -> get_local(a,μ),a)
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,<:LocalReduction{B,C,<:KroneckerReduction}},
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix
  ) where {A,B,C}

  @check a.touched[1] "Primal field not defined"
  tol = RBSteady.get_supr_tol(red)
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  a_primal_loc = local_vals(a_primal)
  for j in eachindex(a_primal_loc)
    pj_space = a_primal_loc[j].projection_space
    pj_time = a_primal_loc[j].projection_time
    for i = eachindex(a_dual)
      if a.touched[i]
        a_dual_i = local_vals(a_dual[i])
        dij_space = get_basis_space(a_dual_i[j])
        C_primal_dual_i = supr_matrix[Block(1,i+1)]
        supr_space_i = H_primal \ C_primal_dual_i * dij_space
        pj_space = union_bases(pj_space,supr_space_i,H_primal)

        dij_time = get_basis_time(a_dual_i[j])
        pj_time = time_enrichment(pj_time,dij_time;tol)
      end
    end
    a_primal_loc[j] = KroneckerProjection(pj_space,pj_time)
  end
  a[1] = RBSteady.local_proj_to_proj(a_primal,a_primal_loc)
  return
end

function RBSteady._cluster(r::GenericTransientRealization,inds::AbstractVector)
  params = RBSteady._cluster(get_params(r),inds)
  times = get_times(r)
  GenericTransientRealization(params,times,r.t0)
end

function RBSteady._cluster(s::TransientSnapshotsWithIC,inds::AbstractVector)
  initial_data = view(s.initial_data,:,inds)
  snaps = RBSteady._cluster(s.snaps,inds)
  TransientSnapshotsWithIC(initial_data,snaps)
end
