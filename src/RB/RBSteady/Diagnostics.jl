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
function hr_error(
  op::RBOperator,
  res::AbstractSnapshots,
  jac::AbstractSnapshots,
  μ::AbstractRealisation
  )

  err_res = hr_error_res(op,res,μ)
  err_jac = hr_error_jac(op,jac,μ)
  return err_res,err_jac
end

function hr_error(
  op::LinearNonlinearRBOperator,
  res::Tuple,
  jac::Tuple,
  μ::AbstractRealisation
  )

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
  res::AbstractSnapshots,
  μ::AbstractRealisation
  )

  b_trian = residual(op,res,μ)
  data = get_all_data(res)
  map(get_contributions(b_trian.hypred)) do h_t
    hr_data = get_all_data(h_t)
    compute_relative_error(data,hr_data)
  end |> Tuple
end

function Algebra.residual(
  op::RBOperator,
  res::AbstractSnapshots,
  μ::AbstractRealisation
  )

  rhs = get_rhs(op)
  test = get_test(op)

  fecache = contribution(get_domains(rhs)) do trian
    interp = get_interpolation(rhs[trian])
    rows = get_interpolation_rows(interp)
    isnothing(rows) ? get_param_data(res) : _get_at_domain(res,rows)
  end

  cache = allocate_hrtrian_cache(rhs,test,μ)
  b = HRParamArrayTrian(fecache,cache.coeff,cache.hypred)
  interpolate!(b,rhs,test)
  return b
end

function hr_error_jac(
  op::RBOperator,
  jac::AbstractSnapshots,
  μ::AbstractRealisation
  )

  A_trian = jacobian(op,jac,μ)
  test = get_test(op)
  trial = get_trial(op)
  Ψ_test = get_basis(get_reduced_subspace(test))
  Ψ_trial = get_basis(get_reduced_subspace(trial))
  np = num_params(μ)

  # Project FOM Jacobian to RB space: (n_rb_test,n_params,n_rb_trial)
  Â_fom = galerkin_projection(Ψ_test,get_param_data(jac),Ψ_trial)
  # Permute to (n_rb_test,n_rb_trial,n_params) to match hypred layout
  Â_fom_p = permutedims(Â_fom,(1,3,2))

  map(get_contributions(A_trian.hypred)) do h_t
    hr_data = get_all_data(h_t)
    mean(1:np) do i
      compute_relative_error(vec(view(Â_fom_p,:,:,i)),vec(view(hr_data,:,:,i)))
    end
  end |> Tuple
end

function Algebra.jacobian(
  op::RBOperator,
  jac::AbstractSnapshots,
  μ::AbstractRealisation
  )

  lhs = get_lhs(op)
  test = get_test(op)
  trial = get_trial(op)

  fecache = contribution(get_domains(lhs)) do trian
    interp = get_interpolation(lhs[trian])
    rows = get_interpolation_rows(interp)
    isnothing(rows) ? get_param_data(jac) :
      _get_at_domain(jac,(rows,get_interpolation_cols(interp)))
  end

  cache = allocate_hrtrian_cache(lhs,trial,test,μ)
  A = HRParamArrayTrian(fecache,cache.coeff,cache.hypred)
  interpolate!(A,lhs,trial,test)
  return A
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

function plot_errors(dir,tols,perfs::AbstractVector{<:ROMPerformance})
  errs = map(get_error,perfs)
  n = length(first(errs))
  errvec = map(i -> getindex.(errs,i),1:n)
  labvec = n==1 ? "Error" : ["Error $i" for i in 1:n]

  file = joinpath(dir,"convergence.png")
  p = plot(tols,tols,lw=3,label="Tol.")
  scatter!(tols,errvec,lw=3,label=labvec)
  plot!(xscale=:log10,yscale=:log10)
  xlabel!("Tolerance")
  ylabel!("Error")
  title!("Average relative error")
  savefig(p,file)
end
