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

function allocate_dcontribution(
  a::AffineContribution,
  r::AbstractRealisation
  )


  fecache = allocate_coefficient(a,r)
  coeff = allocate_coefficient(a,r)
  hypred = contribution(get_domains(a)) do trian
    allocate_hyper_reduction(a[trian],r)
  end
  DiagnosticsContribution(fecache,coeff,hypred)
end

function allocate_diagnostic_residual(nlop::GenericParamNonlinearOperator,u)
  rhs = get_rhs(nlop.op) 
  allocate_dcontribution(rhs,nlop.μ)
end

function allocate_diagnostic_jacobian(nlop::GenericParamNonlinearOperator,u)
  lhs = get_lhs(nlop.op)
  allocate_dcontribution(lhs,nlop.μ)
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

  diagnostic_interpolate!(b,rhs)
end

function diagnostic_residual!(b,nlop::GenericParamNonlinearOperator,u)
  diagnostic_residual!(b,nlop.op,nlop.μ,u,nlop.paramcache)
end

function diagnostic_residual(nlop::NonlinearParamOperator,u)
  b = allocate_diagnostic_residual(nlop,u)
  diagnostic_residual!(b,nlop,u)
  b
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

  diagnostic_interpolate!(A,lhs)
end

function diagnostic_jacobian!(A,nlop::GenericParamNonlinearOperator,u)
  diagnostic_jacobian!(A,nlop.op,nlop.μ,u,nlop.paramcache)
end

function diagnostic_jacobian(nlop::NonlinearParamOperator,u)
  A = allocate_diagnostic_jacobian(nlop,u)
  diagnostic_jacobian!(A,nlop,u)
  A
end

function diagnostic_interpolate!(
  b::DiagnosticsContribution,
  a::AffineContribution
  )

  for (ât,ft,ct,ht) in zip(
    get_contributions(a),
    get_contributions(b.fecache),
    get_contributions(b.coeff),
    get_contributions(b.hypred)
    )

    interpolate!(ht,ct,ât,ft)
  end
end

"""
    struct RBDiagnostics
      offline::Dict{String,Any}
      online::Dict{String,Any}
    end

Container for ROM diagnostics, each phase stored as a `Dict{String,Any}`.

Every dict always contains a `"tols"` key mapping to a `Vector{Float64}` of
tolerances sorted in decreasing order.  All other keys map to a `Vector` of
the same length, with one entry per tolerance.

**Offline** (structural) keys are derived by flattening the `offline_diagnostics`
named tuple:
- `"state dim"`, `"state factor"` — basis size and compression factor
- `"rhs dim"` — `Vector{Tuple}`, one `K`-tuple per tolerance (one integer per
  triangulation)
- `"lhs dim"` — same for the Jacobian contributions
- For `LinearNonlinearRBOperator`: `"lin_rhs dim"`, `"nlin_lhs dim"`, etc.

**Online** keys:
- `"projection_error"` — `Vector{Float64}`
- `"hr_error_res"` — `Vector{Tuple}`, one `K`-tuple per tolerance
- `"hr_error_jac"` — `Vector{Tuple}`, one `K`-tuple per tolerance
"""
struct RBDiagnostics
  offline::Dict{String,Any}
  online::Dict{String,Any}
end

"""
    rom_diagnostics(dir,rbsolver,feop,args...;label="online",kwargs...)
        -> RBDiagnostics

Scans every immediate sub-directory of `dir` whose name parses as a `Float64`
tolerance, loads the corresponding RB operator, and computes both offline
(structural) and online (accuracy) diagnostics using the snapshots stored in
`dir` under `label`.

Returns an [`RBDiagnostics`](@ref) object whose `offline` and `online` fields
are `Dict{String,Any}` sorted by decreasing tolerance (coarsest model first),
with all scalar fields flattened into named `Vector` entries.
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
    err_res,err_jac = hr_error(rbop,res,jac,s)
    push!(online_entries,(tol=tol,diagnostics=(
      projection_error=proj_err,
      hr_error_res=err_res,
      hr_error_jac=err_jac,
    )))
  end

  sort!(offline_entries,by=e->e.tol,rev=true)
  sort!(online_entries,by=e->e.tol,rev=true)
  RBDiagnostics(
    _entries_to_dict(offline_entries),
    _entries_to_dict(online_entries),
  )
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
  (dim=n,factor=N./n)
end

"""
    hr_diagnostics(hrproj::HRProjection) -> NamedTuple

Returns `(dim=n,)` for a single HR projection.
"""
function hr_diagnostics(hrproj::HRProjection)
  n = num_reduced_dofs(hrproj)
  (dim=n,)
end

function hr_diagnostics(hrproj::BlockHRProjection{N}) where N
  s = size(hrproj)
  array = Array{NamedTuple,N}(undef,s)
  for i in eachindex(hrproj)
    if hrproj.touched[i]
      array[i] = hr_diagnostics(hrproj.array[i])
    end
  end
  ArrayBlock(array,hrproj.touched)
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
function hr_error(op::RBOperator,res,jac,s)
  μ = get_realisation(s)
  u = get_param_data(s)
  err_res = hr_error_res(op,res,μ,u)
  err_jac = hr_error_jac(op,jac,μ,u)
  return err_res,err_jac
end

function hr_error(op::RBOperator{<:LinearParamEq},res,jac,s)
  μ = get_realisation(s)
  u = get_param_data(s)|> similar
  fill!(u,zero(eltype2(u)))
  err_res = hr_error_res(op,res,μ,u)
  err_jac = hr_error_jac(op,jac,μ,u)
  return err_res,err_jac
end

function hr_error(op::LinearNonlinearRBOperator,res,jac,s)
  res_lin,res_nlin = res
  jac_lin,jac_nlin = jac
  op_lin = get_linear_operator(op)
  op_nlin = get_nonlinear_operator(op)
  (err_res_lin,err_jac_lin) = hr_error(op_lin,res_lin,jac_lin,s)
  (err_res_nlin,err_jac_nlin) = hr_error(op_nlin,res_nlin,jac_nlin,s)
  return (err_res_lin,err_res_nlin),(err_jac_lin,err_jac_nlin)
end

function hr_error_res(
  op::RBOperator,
  res::ArrayContribution,
  μ::AbstractRealisation,
  u
  )

  test = get_test(op)
  rhs = get_rhs(op)
  nlop = parameterise(op,μ)
  red_res = diagnostic_residual(nlop,u)  

  err = ()
  for (res_t,a_t,fecache_t,hypred_t) in zip(
    get_contributions(res),
    get_contributions(rhs),
    get_contributions(red_res.fecache),
    get_contributions(red_res.hypred)
    )

    err = (err...,hr_error_res(test,res_t,a_t,fecache_t,hypred_t))
  end 
  
  return err
end

function hr_error_jac(
  op::RBOperator,
  jac::ArrayContribution,
  μ::AbstractRealisation,
  u
  )

  test  = get_test(op)
  trial = get_trial(op)
  lhs = get_lhs(op)
  nlop = parameterise(op,μ)
  red_jac = diagnostic_jacobian(nlop,u)

  err = ()
  for (jac_t,a_t,fecache_t,hypred_t) in zip(
    get_contributions(jac),
    get_contributions(lhs),
    get_contributions(red_jac.fecache),
    get_contributions(red_jac.hypred)
    )

    err = (err...,hr_error_jac(trial,test,jac_t,a_t,fecache_t,hypred_t))
  end 
  
  return err
end

function hr_error_res(
  test::SingleFieldRBSpace,
  res::Snapshots,
  a::HRProjection,
  fecache::AbstractParamVector,
  hypred::AbstractParamVector
  )
  
  msg = "fecache mismatch at DEIM interpolation rows"
  Φ_test = get_basis(get_reduced_subspace(test))  
  rows = get_interpolation_rows(get_interpolation(a))
  @check isapprox(get_all_data(fecache),get_all_data(res)[rows,:];rtol=1e-8) msg

  b̂ = galerkin_projection(Φ_test,get_all_data(res))

  μ = get_realisation(res)
  i = VectorDofMap(size(b̂,1))
  b̂snaps = Snapshots(b̂,i,μ)
  hrb̂snaps = Snapshots(get_all_data(hypred),i,μ)

  compute_relative_error(b̂snaps,hrb̂snaps)
end

function hr_error_jac(
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace,
  jac::Snapshots,
  a::HRProjection,
  fecache::AbstractParamVector,
  hypred::AbstractParamMatrix
  )
  
  msg = "fecache mismatch at DEIM (row,col) pairs"
  Φ_trial = get_basis(get_reduced_subspace(trial))  
  Φ_test = get_basis(get_reduced_subspace(test))  

  rows = get_interpolation_rows(get_interpolation(a))
  cols = get_interpolation_cols(get_interpolation(a))
  sparsity = get_sparsity(get_dof_map(jac))
  inds = sparsify_split_indices(rows,cols,sparsity)
  @check isapprox(get_all_data(fecache),get_all_data(jac)[inds,:];rtol=1e-8) msg

  μ = get_realisation(jac)
  Â = galerkin_projection(Φ_test,recast(jac),Φ_trial)
  Â = reshape(permutedims(Â,(1,3,2)),:,num_params(μ))
  hrÂ = reshape(get_all_data(hypred),:,num_params(μ))

  i = VectorDofMap(size(Â,1))
  Âsnaps = Snapshots(Â,i,μ)
  hrÂsnaps = Snapshots(hrÂ,i,μ)

  compute_relative_error(Âsnaps,hrÂsnaps)
end

function hr_error_res(
  test::MultiFieldRBSpace,
  res::BlockSnapshots,
  a::BlockHRProjection,
  fecache::VectorBlock,
  hypred::BlockParamVector
  )
  
  @check res.touched == fecache.touched
  error = zeros(size(res))
  for i in eachindex(res)
    if res.touched[i]
      error[i] = hr_error_res(test[i],res[i],a[i],fecache.array[i],hypred.data[i])
    end
  end
  error
end

function hr_error_jac(
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace,
  jac::BlockSnapshots,
  a::BlockHRProjection,
  fecache::MatrixBlock,
  hypred::BlockParamMatrix
  )
  
  @check jac.touched == fecache.touched
  error = zeros(size(jac))
  for i in axes(jac,1), j in axes(jac,2)
    if jac.touched[i,j]
      error[i,j] = hr_error_jac(trial[j],test[i],jac[i,j],a[i,j],fecache.array[i,j],hypred.data[i,j])
    end
  end
  error
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

function save_residuals(dir,feop,res;label="")
  save(dir,res;label=_get_label(label,residuals_label))
end

function save_jacobians(dir,feop,jac;label="")
  save(dir,jac;label=_get_label(label,jacobians_label))
end

for f in (:save_residuals,:save_jacobians)
  @eval begin
    function $f(dir,feop::LinearNonlinearParamOperator,resjac::Tuple;label="")
      @assert length(resjac) == 2
      $f(dir,get_linear_operator(feop),resjac[1];label=_get_label(label,linear_label))
      $f(dir,get_nonlinear_operator(feop),resjac[2];label=_get_label(label,nonlinear_label))
      return
    end
  end
end

function load_residuals(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_res(feop);label=_get_label(label,residuals_label))
end

function load_jacobians(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_jac(feop);label=_get_label(label,jacobians_label))
end

for f in (:load_residuals,:load_jacobians)
  @eval begin
    function $f(dir,feop::LinearNonlinearParamOperator;label="")
      (
        $f(dir,get_linear_operator(feop);label=_get_label(label,linear_label)),
        $f(dir,get_nonlinear_operator(feop);label=_get_label(label,nonlinear_label)),
      )
    end
  end
end

function load_residuals(dir,_rbsolver,feop,fesnaps)
  try
    load_residuals(dir,feop;label=residuals_label)
  catch
    rbsolver = set_params(_rbsolver,nparams=num_params(fesnaps))
    res = residual_snapshots(rbsolver,feop,fesnaps)
    save_residuals(dir,feop,res)
    res
  end
end

function load_jacobians(dir,_rbsolver,feop,fesnaps)
  try
    load_jacobians(dir,feop;label=jacobians_label)
  catch
    rbsolver = set_params(_rbsolver;nparams=num_params(fesnaps))
    jac = jacobian_snapshots(rbsolver,feop,fesnaps)
    save_jacobians(dir,feop,jac)
    jac
  end
end

function load_problem_snapshots(dir,rbsolver,feop,args...;label="online",kwargs...)
  s = load_snapshots(dir,rbsolver,feop,args...;label=label,kwargs...)
  jac = load_jacobians(dir,rbsolver,feop,s)
  res = load_residuals(dir,rbsolver,feop,s)
  return s,jac,res
end

# utils 

function set_params(red::AffineReduction;kwargs...)
  red
end

function set_params(red::PODReduction;nparams::Int)
  PODReduction(red.red_style,red.norm_style,nparams)
end

function set_params(red::TTSVDReduction;nparams::Int)
  TTSVDReduction(red.red_style,red.norm_style,nparams)
end

function set_params(red::LocalReduction;kwargs...)
  LocalReduction(set_params(red.reduction;kwargs...),red.ncentroids)
end

function set_params(red::SupremizerReduction;kwargs...)
  SupremizerReduction(set_params(red.reduction;kwargs...),red.supr_op,red.supr_tol)
end

function set_params(red::MDEIMHyperReduction;kwargs...)
  MDEIMHyperReduction(set_params(red.reduction;kwargs...))
end

function set_params(red::SOPTHyperReduction;kwargs...)
  SOPTHyperReduction(set_params(red.reduction;kwargs...))
end

function set_params(red::RBFHyperReduction;kwargs...)
  RBFHyperReduction(set_params(red.reduction;kwargs...),red.strategy)
end

function set_params(rbsolver;kwargs...)
  fesolver = get_fe_solver(rbsolver)
  state_reduction = set_params(get_state_reduction(rbsolver);kwargs...)
  residual_reduction = set_params(get_residual_reduction(rbsolver);kwargs...)
  jacobian_reduction = set_params(get_jacobian_reduction(rbsolver);kwargs...)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function _entries_to_dict(entries::Vector{<:NamedTuple})
  d = Dict{String,Any}()
  isempty(entries) && return d
  d["tols"] = [e.tol for e in entries]
  _unpack_into_dict!(d,"",map(e -> e.diagnostics,entries))
  return d
end

function _unpack_into_dict!(d::Dict,prefix::String,vals::Vector)
  isempty(vals) && return
  v1 = first(vals)
  if v1 isa NamedTuple
    for k in keys(v1)
      _unpack_into_dict!(d,_diagkey(prefix,string(k)),map(v -> v[k],vals))
    end
  elseif v1 isa Tuple && !isempty(v1) && first(v1) isa NamedTuple
    for k in keys(first(v1))
      _unpack_into_dict!(
        d,_diagkey(prefix,string(k)),
        map(v -> Tuple(vt[k] for vt in v),vals),
      )
    end
  else
    d[prefix] = vals
  end
end

_diagkey(prefix,key) = isempty(prefix) ? key : "$prefix $key"