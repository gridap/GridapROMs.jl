function residual_snapshots(
  solver::RBSolver,
  op::UnCommonParamOperator,
  s::AbstractSnapshots)

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres)
  r_res = get_realization(sres)
  b = residual(op,r_res,us_res)
  ib = get_dof_map(op)
  return Snapshots(b,ib,r_res)
end

function residual_snapshots(
  solver::RBSolver,
  op::UnCommonParamOperator{LinearParamEq},
  s::AbstractSnapshots)

  sres = select_snapshots(s,res_params(solver))
  us_res = get_param_data(sres) |> similar
  fill!(us_res,zero(eltype2(us_res)))
  r_res = get_realization(sres)
  b = residual(op,r_res,us_res)
  ib = get_dof_map(op)
  return Snapshots(b,ib,r_res)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::UnCommonParamOperator,
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac)
  r_jac = get_realization(sjac)
  A = jacobian(op,r_jac,us_jac)
  sparsity = get_common_sparsity(A)
  iA = get_sparse_dof_map(sparsity,op)
  return Snapshots(A,iA,r_jac)
end

function jacobian_snapshots(
  solver::RBSolver,
  op::UnCommonParamOperator{LinearParamEq},
  s::AbstractSnapshots)

  fesolver = get_fe_solver(solver)
  sjac = select_snapshots(s,jac_params(solver))
  us_jac = get_param_data(sjac) |> similar
  fill!(us_jac,zero(eltype2(us_jac)))
  r_jac = get_realization(sjac)
  A = jacobian(op,r_jac,us_jac)
  iA = get_sparse_dof_map(sparsity,op)
  return Snapshots(A,iA,r_jac)
end
