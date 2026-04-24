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

  feop = get_fe_operator(op)
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

"""
    results_table(dir,feop) -> Vector{NamedTuple}

Scans every immediate sub-directory of `dir` whose name parses as a `Float64`
(i.e. a tolerance value produced by [`run_test`](@ref)).  For each such
sub-directory that contains a saved RB operator the function:

1. loads the operator with `load_operator`,
2. calls `rom_diagnostics` to extract basis dimension and compression factors
   for the state space and for every hyper-reduced contribution,
3. collects everything into a named tuple.

The returned vector is sorted in decreasing order of tolerance (coarsest first).
"""
function results_table(dir::String,feop::ParamOperator)
  entries = NamedTuple[]

  for name in sort(readdir(dir))
    subdir = joinpath(dir,name)
    isdir(subdir) || continue
    tol = tryparse(Float64,name)
    isnothing(tol) && continue

    # check that at least the test-basis file exists before trying to deserialize
    isfile(joinpath(subdir,"basis_test.jld")) || continue

    rbop = try
      load_operator(subdir,feop)
    catch e
      @warn "Could not load operator from $subdir: $e"
      continue
    end

    diag = rom_diagnostics(rbop)
    push!(entries,(tol=tol,diagnostics=diag))
  end

  sort!(entries,by=e->e.tol,rev=true)
  return entries
end

"""
    rom_diagnostics(rbop::RBOperator) -> NamedTuple

Returns structural measures of the RB operator:
- `state`: `(dim = n, factor = Nₕ / n)` for the trial/test reduction
- `res`: tuple of `(dim,factor)` per residual triangulation
- `jac`: tuple of `(dim,factor)` per jacobian triangulation
"""
function rom_diagnostics(rbop::RBOperator)
  trial = get_trial(rbop)
  state = projection_diagnostics(trial)
  res = Tuple(_hr_measures(v) for v in get_contributions(get_rhs(rbop)))
  jac = Tuple(_hr_measures(v) for v in get_contributions(get_lhs(rbop)))
  (state=state,res=res,jac=jac)
end

function rom_diagnostics(rbop::LinearNonlinearRBOperator)
  state = _space_measures(get_test(rbop))
  op_lin = get_linear_operator(rbop)
  op_nlin = get_nonlinear_operator(rbop)
  res_lin = Tuple(_hr_measures(v) for v in get_contributions(get_rhs(op_lin)))
  res_nlin = Tuple(_hr_measures(v) for v in get_contributions(get_rhs(op_nlin)))
  jac_lin = Tuple(_hr_measures(v) for v in get_contributions(get_lhs(op_lin)))
  jac_nlin = Tuple(_hr_measures(v) for v in get_contributions(get_lhs(op_nlin)))
  (state=state,
   res=(linear=res_lin,nonlinear=res_nlin),
   jac=(linear=jac_lin,nonlinear=jac_nlin))
end

function _space_measures(r::RBSpace)
  N = num_fe_dofs(r)
  n = num_reduced_dofs(r)
  (dim=n,factor=N/n)
end

function _hr_measures(c::AffineContribution)
  Tuple(_hr_measures(v) for v in get_contributions(c))
end

function _hr_measures(hrproj::HRProjection)
  n_modes = num_reduced_dofs(hrproj)
  n_left  = num_reduced_dofs_left_projector(hrproj)
  (dim=n_modes,factor=n_left/n_modes)
end

function _hr_measures(hrproj::BlockHRProjection{N}) where N
  Tuple(hrproj.touched[i] ? _hr_measures(hrproj.array[i]) : nothing
        for i in eachindex(hrproj.touched))
end