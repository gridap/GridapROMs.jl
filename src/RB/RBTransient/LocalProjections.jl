function RBSteady.local_vals(a::KroneckerProjection)
  map(KroneckerProjection,local_vals(a.projection_space),local_vals(a.projection_time))
end

function RBSteady.local_vals(a::SequentialProjection)
  map(SequentialProjection,local_vals(a.projection))
end

RBSteady.get_clusters(a::KroneckerProjection) = get_clusters(a.projection_space)
RBSteady.get_clusters(a::SequentialProjection) = get_clusters(a.projection)

RBSteady.get_local(a,r::TransientRealisation) = get_local(a,get_params(r))
RBSteady.get_local(a,μt::Tuple{Any,Any}) = get_local(a,first(μt))

function RBSteady.get_local(a::KroneckerProjection,μ::AbstractVector)
  KroneckerProjection(get_local(a.projection_space,μ),get_local(a.projection_time,μ))
end

function RBSteady.get_local(a::SequentialProjection,μ::AbstractVector)
  SequentialProjection(get_local(a.projection,μ))
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,D,<:LocalReduction{B,C,<:KroneckerReduction}},
  a::BlockProjection,
  norm_matrix::BlockMatrix,
  supr_matrix::BlockMatrix
  ) where {A,B,C,D}

  tol = RBSteady.get_supr_tol(red)
  a_primal,a_dual... = a.array
  X_primal = norm_matrix[Block(1,1)]
  H_primal = symcholesky(X_primal)
  a_primal_loc = local_vals(a_primal)
  for j in eachindex(a_primal_loc)
    pj_space = a_primal_loc[j].projection_space
    pj_time = a_primal_loc[j].projection_time
    for i = eachindex(a_dual)
      a_dual_i = local_vals(a_dual[i])
      dij_space = get_basis_space(a_dual_i[j])
      C_primal_dual_i = supr_matrix[Block(1,i+1)]
      supr_space_i = supremizers(H_primal,C_primal_dual_i,dij_space)
      pj_space = union_bases(pj_space,supr_space_i,H_primal)

      dij_time = get_basis_time(a_dual_i[j])
      pj_time = time_enrichment(pj_time,dij_time;tol)
    end
    a_primal_loc[j] = KroneckerProjection(pj_space,pj_time)
  end
  a[1] = RBSteady.local_proj_to_proj(a_primal,a_primal_loc)
  return
end

function RBSteady.enrich!(
  red::SupremizerReduction{A,D,<:LocalReduction{B,C,<:SequentialReduction}},
  a::BlockProjection,
  norm_matrix::BlockRankTensor,
  supr_matrix::BlockRankTensor;
  kwargs...
  ) where {A,B,C,D}

  red′ = SupremizerReduction(LocalReduction(red.reduction.reduction.reduction),red.coupling,red.supr_tol)
  enrich!(red′,a,norm_matrix,supr_matrix;kwargs...)
end

function RBSteady._cluster(r::GenericTransientRealisation,inds::AbstractVector)
  params = RBSteady._cluster(get_params(r),inds)
  times = get_times(r)
  GenericTransientRealisation(params,times,r.t0)
end

function RBSteady._cluster(s::TransientSnapshotsWithIC,inds::AbstractVector)
  data_inds(d) = ParamDataStructures.select_param_data(d,inds)
  initial_param_data = map(data_inds,s.initial_param_data)
  snaps = RBSteady._cluster(s.snaps,inds)
  TransientSnapshotsWithIC(initial_param_data,snaps)
end

function RBSteady._cluster(s::TransientBlockSnapshots{N},inds::AbstractVector) where N
  array = map(sj -> RBSteady._cluster(sj,inds),blocks(s))
  nt = num_times(get_realisation(s))
  pdata = RBSteady._cluster(s.param_data,inds,nt)
  return BlockSnapshots(array,pdata)
end

function RBSteady._cluster(a::StoredParamData,inds::AbstractVector,nt::Int)
  param_data = ParamDataStructures.select_param_data(a.param_data,inds,1:nt)
  param_data0 = map(d0 -> RBSteady._cluster(d0,inds),a.param_data0)
  StoredParamData(param_data,param_data0)
end
