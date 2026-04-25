"""
    struct DiagnosticsContribution{A,B,C}
      fecache::A
      coeff::B
      hypred::C
    end

Diagnostic counterpart of [`HRParamArray`](@ref). Unlike `HRParamArray`, which
accumulates hyper-reduced contributions across triangulations into a single
reduced-dimension array, `DiagnosticsContribution` keeps one per-triangulation entry
in `hypred::C`, where `C` is either an `ArrayContribution` (steady) or a
`TupOfArrayContribution` (transient Jacobians). Each entry stores the
reconstruction of the HR operator contribution from that triangulation,
expanded back to a high-dimensional (FE or RB) space so that it can be
directly compared with full-order snapshots.
"""
struct DiagnosticsContribution{A,B,C}
  fecache::A
  coeff::B
  hypred::C
end

function allocate_diagnostic_residual(
  a::AffineContribution,
  test::RBSpace,
  r::AbstractRealisation
  )

  N = num_fe_dofs(test)
  np = num_params(r)
  fecache = allocate_coefficient(a,r)
  coeff = allocate_coefficient(a,r)
  b = zeros(N)
  hypred = contribution(get_domains(a)) do _
    parameterise(similar(b),np)
  end
  DiagnosticsContribution(fecache,coeff,hypred)
end

function allocate_diagnostic_residual(nlop::NonlinearOperator,u)
  op = get_fe_operator(nlop)
  rhs = get_rhs(op)
  test = get_test(op)
  r = get_realisation(nlop)
  allocate_diagnostic_residual(rhs,test,r)
end

function allocate_diagnostic_jacobian(
  a::AffineContribution,
  trial::RBSpace,
  test::RBSpace,
  r::AbstractRealisation
  )

  np = num_params(r)
  fecache = allocate_coefficient(a,r)
  coeff = allocate_coefficient(a,r)
  A = get_background_matrix(get_sparsity(trial,test))
  hypred = contribution(get_domains(a)) do _
    parameterise(similar(A),np)
  end
  DiagnosticsContribution(fecache,coeff,hypred)
end

function allocate_diagnostic_jacobian(nlop::NonlinearOperator,u)
  op = get_fe_operator(nlop)
  lhs = get_lhs(op)
  trial = get_trial(op)
  test = get_test(op)
  r = get_realisation(nlop)
  allocate_diagnostic_jacobian(lhs,trial,test,r)
end

function diagnostic_residual!(
  b::DiagnosticsContribution,
  op::SplitRBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  trian_res = get_domains_res(op)
  rhs = get_rhs(op)
  res = get_res(op)
  dc = res(r,uh,v)

  for strian in trian_res
    b_strian = b.fecache[strian]
    rhs_strian = get_interpolation(rhs[strian])
    vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
    assemble_hr_vector_add!(b_strian,vecdata...)
  end

  diagnostic_interpolate!(b,test,rhs)
end

function diagnostic_residual!(b,nlop::NonlinearOperator,u)
  diagnostic_residual!(b,nlop.op,nlop.r,u,nlop.paramcache)
end

function diagnostic_residual(nlop::NonlinearOperator,u)
  b = allocate_diagnostic_residual(nlop,u)
  diagnostic_residual!(b,nlop,u)
end

function diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::SplitRBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  trian_jac = get_domains_jac(op)
  lhs = get_lhs(op)
  jac = get_jac(op)
  dc = jac(r,uh,du,v)

  for strian in trian_jac
    A_strian = A.fecache[strian]
    lhs_strian = get_interpolation(lhs[strian])
    matdata = collect_cell_hr_matrix(trial,test,dc,strian,lhs_strian)
    assemble_hr_matrix_add!(A_strian,matdata...)
  end

  diagnostic_interpolate!(A,trial,test,lhs)
end

function diagnostic_jacobian!(A,nlop::NonlinearOperator,u)
  diagnostic_jacobian!(A,nlop.op,nlop.r,u,nlop.paramcache)
end

function diagnostic_jacobian(nlop::NonlinearOperator,u)
  A = allocate_diagnostic_jacobian(nlop,u)
  diagnostic_jacobian!(A,nlop,u)
end

function diagnostic_interpolate!(
  b::DiagnosticsContribution,
  test::RBSpace,
  a::AffineContribution
  )
  
  for (ât,ft,ct,ht) in zip(
    get_contributions(a),
    get_contributions(b.fecache),
    get_contributions(b.coeff),
    get_contributions(b.hypred)
    ) 
    
    at = inv_galerkin_projection(test,ât)   
    interpolate!(ht,ct,at,ft)
  end
end

function diagnostic_interpolate!(
  A::DiagnosticsContribution,
  trial::RBSpace,
  test::RBSpace,
  a::AffineContribution
  )

  for (ât,ft,ct,ht) in zip(
    get_contributions(a),
    get_contributions(A.fecache),
    get_contributions(A.coeff),
    get_contributions(A.hypred)
    ) 
    
    at = inv_galerkin_projection(test,ât,trial)   
    interpolate!(ht,ct,at,ft)
  end
end

"""
    struct RBDiagnostics
      offline
      online
    end

Container for ROM diagnostics split into two phases.

- `offline`: vector of `(tol, diagnostics)` named tuples, one per
  tolerance sub-directory.  Each `diagnostics` entry is a named tuple
  with the structural properties of the RB operator: basis dimension,
  compression factors for every hyper-reduced triangulation contribution.

- `online`: vector of `(tol, diagnostics)` named tuples.  Each
  `diagnostics` entry is a named tuple with
  - `projection_error`: average relative error of the RB projection of
    the solution snapshots
  - `hr_error_res`: tuple of per-triangulation HR errors for the residual
  - `hr_error_jac`: tuple of per-triangulation HR errors for the Jacobian
"""
struct RBDiagnostics
  offline
  online
end

"""
    rom_diagnostics(dir,rbsolver,feop,args...;label="online",kwargs...)
        -> RBDiagnostics

Scans every immediate sub-directory of `dir` whose name parses as a `Float64`
tolerance, loads the corresponding RB operator, and computes both offline
(structural) and online (accuracy) diagnostics using the snapshots stored in
`dir` under `label`.

Returns an [`RBDiagnostics`](@ref) object whose `offline` and `online` fields
are vectors sorted by decreasing tolerance (coarsest model first).
"""
function rom_diagnostics(
  dir::String,
  rbsolver::RBSolver,
  feop::ParamOperator,
  args...;
  label="online",
  kwargs...
  )

  s,jac,res = load_problem_snapshots(dir,rbsolver,feop,args...;label,kwargs...)
  μ = get_realisation(s)

  offline_entries = NamedTuple[]
  online_entries = NamedTuple[]

  for name in sort(readdir(dir))
    subdir = joinpath(dir,name)
    isdir(subdir) || continue
    tol = tryparse(Float64,name)
    isnothing(tol) && continue

    rbop = try
      load_operator(subdir,feop)
    catch e
      @warn "Could not load operator from $subdir: $e"
      continue
    end

    push!(offline_entries,(tol=tol,diagnostics=offline_diagnostics(rbop)))

    proj_err = projection_error(rbsolver,rbop,s)
    err_res,err_jac = hr_error(rbop,res,jac,μ)
    push!(online_entries,(tol=tol,diagnostics=(
      projection_error=proj_err,
      hr_error_res=err_res,
      hr_error_jac=err_jac,
    )))
  end

  sort!(offline_entries,by=e->e.tol,rev=true)
  sort!(online_entries,by=e->e.tol,rev=true)
  RBDiagnostics(offline_entries,online_entries)
end

function offline_diagnostics(op::RBOperator)
  (
    state=projection_diagnostics(get_trial(op)),
    rhs=hr_diagnostics(get_rhs(op)),
    lhs=hr_diagnostics(get_lhs(op)),
  )
end

function offline_diagnostics(op::LinearNonlinearRBOperator)
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  (
    state=projection_diagnostics(get_trial(op)),
    lin_rhs=hr_diagnostics(get_rhs(op_lin)),
    lin_lhs=hr_diagnostics(get_lhs(op_lin)),
    nlin_rhs=hr_diagnostics(get_rhs(op_nlin)),
    nlin_lhs=hr_diagnostics(get_lhs(op_nlin)),
  )
end

"""
    projection_diagnostics(r::RBSpace) -> NamedTuple

Returns `(dim=n,factor=Nₕ/n)` for the trial/test reduction.
"""
function projection_diagnostics(r::RBSpace)
  N = num_fe_dofs(r)
  n = num_reduced_dofs(r)
  (dim=n,factor=N/n)
end

"""
    hr_diagnostics(hrproj::HRProjection) -> NamedTuple

Returns `(dim=n_modes,factor=n_left/n_modes)` for a single HR projection.
"""
function hr_diagnostics(hrproj::HRProjection)
  n_modes = num_reduced_dofs(hrproj)
  n_left = num_reduced_dofs_left_projector(hrproj)
  (dim=n_modes,factor=n_left/n_modes)
end

function hr_diagnostics(hrproj::BlockHRProjection{N}) where N
  s = size(hproj)
  array = Array{NamedTuple,N}(undef,s)
  for i in eachindex(hproj)
    if hrproj.touched[i]
      array[i] = hr_diagnostics(hproj.array[i])
    end
  end
  ArrayBlock(array,hproj.touched)
end

function hr_diagnostics(c::AffineContribution)
  Tuple(hr_diagnostics(v) for v in get_contributions(c))
end

"""
    projection_error(solver,op,s) -> Number

Average relative error committed by projecting the solution snapshots `s` onto
the RB trial space of `op`.
"""
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

"""
    hr_error(op,res,jac,μ) -> (Tuple,Tuple)

Compute per-triangulation hyper-reduction errors for residuals and Jacobians.

For each triangulation in the HR contributions:
- **Residuals**: the HR reconstruction `Ψ·Φrb·coeff` (in FE space) is compared
  with the full-order snapshot vector using the Euclidean norm.
- **Jacobians**: the HR reconstruction `Φrb·coeff` (in RB space) is compared
  with the Galerkin projection of the FOM Jacobian onto the RB subspace using
  the Frobenius norm; this equals the full-space Frobenius error when the RB
  bases are orthonormal.

Returns `(hr_error_res,hr_error_jac)` where each is a `Tuple` with one
`Float64` per triangulation (mean relative error over parameters).
"""
function hr_error(op::RBOperator,res,jac,μ)
  err_res = hr_error_res(op,res,μ)
  err_jac = hr_error_jac(op,jac,μ)
  return err_res,err_jac
end

function hr_error(op::LinearNonlinearRBOperator,res,jac,μ)
  res_lin,res_nlin = res
  jac_lin,jac_nlin = jac
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  err_res = (hr_error_res(op_lin,res_lin,μ),hr_error_res(op_nlin,res_nlin,μ))
  err_jac = (hr_error_jac(op_lin,jac_lin,μ),hr_error_jac(op_nlin,jac_nlin,μ))
  return err_res,err_jac
end

function hr_error_res(
  op::RBOperator,
  res::ArrayContribution,
  μ::AbstractRealisation,
  u
  )

  nlop = parameterise(op,μ)
  red_res = diagnostic_residual(nlop,u)
  map(get_contributions(res),get_contributions(red_res.hypred)) do b,b̂
    compute_relative_error(get_data(b),get_data(b̂))
  end |> Tuple
end

function hr_error_jac(
  op::RBOperator,
  jac::AbstractSnapshots,
  μ::AbstractRealisation,
  u
  )

  nlop = parameterise(op,μ)
  red_jac = diagnostic_jacobian(nlop,u)
  map(get_contributions(jac),get_contributions(red_jac.hypred)) do A,Â
    compute_relative_error(get_data(A),get_data(Â))
  end |> Tuple
end

function load_snapshots(dir,rbsolver,feop,args...;label="",kwargs...)
  try
    load_snapshots(dir;label)
  catch
    s,stats = solution_snapshots(rbsolver,feop,args...;kwargs...)
    save(dir,s;label)
    save(dir,stats;label)
    s
  end
end

function save_residuals(dir,res;label="")
  save(dir,res;label=_get_label(label,"res"))
end

function save_jacobians(dir,jac;label="")
  save(dir,jac;label=_get_label(label,"jac"))
end

for f in (:save_residuals,:save_jacobians)
  @eval begin
    function $f(dir,resjac::Tuple;label="")
      @assert length(resjac) == 2
      $f(dir,resjac[1];label=_get_label(label,"lin"))
      $f(dir,resjac[2];label=_get_label(label,"nlin"))
      return
    end
  end
end

function load_residuals(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_res(feop);label=_get_label(label,"res"))
end

function load_jacobians(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_jac(feop);label=_get_label(label,"jac"))
end

for f in (:load_residuals,:load_jacobians)
  @eval begin
    function $f(dir,feop::LinearNonlinearParamOperator;label="")
      (
        $f(dir,get_linear_operator(feop);label=_get_label(label,"lin")),
        $f(dir,get_nonlinear_operator(feop);label=_get_label(label,"nlin")),
      )
    end
  end
end

function load_residuals(dir,rbsolver,feop,fesnaps)
  try
    load_snapshots(dir;label="res")
  catch
    res = residual_snapshots(rbsolver,feop,fesnaps)
    save_residuals(dir,res)
    res
  end
end

function load_jacobians(dir,rbsolver,feop,fesnaps)
  try
    load_snapshots(dir;label="jac")
  catch
    jac = jacobian_snapshots(rbsolver,feop,fesnaps)
    save_jacobians(dir,jac)
    jac
  end
end

function load_problem_snapshots(dir,rbsolver,feop,args...;label="online",kwargs...)
  s = load_snapshots(dir,rbsolver,feop,args...;label=label,kwargs...)
  jac = load_jacobians(dir,rbsolver,feop,s)
  res = load_residuals(dir,rbsolver,feop,s)
  return s,jac,res
end