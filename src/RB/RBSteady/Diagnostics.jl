struct RBDiagnostics
  projection_error
  hr_error_res
  hr_error_jac
end

function rom_diagnostics(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractSnapshots,
  jac,
  res
  )

  μ = get_realisation(s)
  nlop = parameterise(op,μ)
  proj_error = projection_error(solver,op,s)
  hr_err_jac,hr_err_res = hr_error(solver,nlop,jac,res)
  RBError(proj_error,hr_err_res,hr_err_jac)
end

function rom_diagnostics(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractSnapshots
  )

  feop = set_domains(get_fe_operator(op))
  jac = jacobian_snapshots(solver,feop,s)
  res = residual_snapshots(solver,feop,s)
  rom_diagnostics(solver,op,s,jac,res)
end

function rom_diagnostics(
  solver::RBSolver,
  op::RBOperator,
  μ::AbstractRealisation,
  args...
  )

  feop = get_fe_operator(op)
  s = solution_snapshots(solver,feop,μ,args...)
  rom_diagnostics(solver,op,s)
end

function projection_error(
  solver::RBSolver,
  op::RBOperator,
  s::AbstractSnapshots
  )

  μ = get_realisation(s)
  feop = get_fe_operator(op)
  trial = get_trial(op)(μ)
  x = get_param_data(s)
  x̂ = project(trial,x)
  ŝ = to_snapshots(op,x̂,μ)
  compute_relative_error(solver,feop,s,ŝ)
end

function hr_error(solver::RBSolver,op::NonlinearParamOperator,A,b)
  trial = get_trial(op)
  test = get_test(op)

  x = zero_initial_guess(op)
  μ = get_realisation(op)
  Â = jacobian(op,x)
  b̂ = residual(op,x)
  
  jac_red = get_jacobian_reduction(solver)
  res_red = get_residual_reduction(solver)
  hr_err_jac = hr_error(jac_red,trial,test,A,Â,μ)
  hr_err_res = hr_error(res_red,test,b,b̂,μ)

  return (hr_err_jac,hr_err_res)
end

function hr_error(
  ::Reduction,
  test::SingleFieldRBSpace,
  b::AbstractParamVector,
  b̂::AbstractParamVector,
  μ::AbstractRealisation
  )

  basis_left = get_basis(test)
  pdata = get_param_data(b)
  data = get_all_data(pdata)
  bproj = galerkin_projection(basis_left,data)
  
  i = VectorDofMap(size(testitem(pdata)))
  b̂snaps = Snapshots(b̂,i,μ)
  bsnaps = Snapshots(bproj,i,μ)

  compute_relative_error(b̂snaps,bsnaps)
end

function hr_error(
  ::Reduction,
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace,
  A::AbstractParamMatrix,
  Â::AbstractParamMatrix,
  μ::AbstractRealisation
  )

  basis_left = get_basis(test)
  pdata = get_param_data(A)
  data = get_all_data(pdata)
  basis_right = get_basis(trial)
  Aproj = galerkin_projection(basis_left,data,basis_right)
  
  i = VectorDofMap(size(testitem(pdata)))
  Âsnaps = Snapshots(Â,i,μ)
  Asnaps = Snapshots(Aproj,i,μ)
  
  compute_relative_error(Âsnaps,Asnaps)
end

# multi-field interface

function hr_error(
  red::Reduction,
  test::MultiFieldRBSpace,
  b::AbstractParamVector,
  b̂::AbstractParamVector,
  μ::AbstractRealisation
  )

  array = map(1:num_fields(test)) do i
    hr_error(red,test[i],b.blocks[i],b̂.blocks[i],μ)
  end
  ArrayBlocks(array,fill(true,size(array)))
end

function hr_error(
  red::Reduction,
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace,
  A::AbstractParamMatrix,
  Â::AbstractParamMatrix,
  μ::AbstractRealisation
  )

  array = map(Iterators.product(1:num_fields(test),1:num_fields(trial))) do (i,j)
    hr_error(red,trial[j],test[i],A.blocks[i,j],Â.blocks[i,j],μ)
  end
  ArrayBlocks(array,fill(true,size(array)))
end
