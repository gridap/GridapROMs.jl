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

function allocate_diagnostic_residual(
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  rhs = get_rhs(op)
  allocate_dcontribution(rhs,r)
end

function allocate_diagnostic_residual(
  op::RBOperator{O,T,<:NoHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  rhs = get_rhs(op) 
  b = allocate_residual(op.op,r,u,paramcache)
  b̂ = allocate_dcontribution(rhs,r)
  DiagnosticsContribution(b,b̂.coeff,b̂.hypred)
end

function allocate_diagnostic_residual(nlop::GenericParamNonlinearOperator,u)
  allocate_diagnostic_residual(nlop.op,nlop.μ,u,nlop.paramcache)
end

function allocate_diagnostic_jacobian(
  op::RBOperator,
  r::Realisation,
  u::AbstractVector,
  paramcache
  )

  lhs = get_lhs(op)
  allocate_dcontribution(lhs,r)
end

function allocate_diagnostic_jacobian(
  op::RBOperator{O,T,B,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  lhs = get_lhs(op)
  A = allocate_jacobian(op.op,r,u,paramcache)
  Â = allocate_dcontribution(lhs,r)
  DiagnosticsContribution(A,Â.coeff,Â.hypred)
end

function allocate_diagnostic_jacobian(nlop::GenericParamNonlinearOperator,u)
  allocate_diagnostic_jacobian(nlop.op,nlop.μ,u,nlop.paramcache)
end

function diagnostic_residual!(
  b::DiagnosticsContribution,
  op::SplitReducedOperator,
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
    assemble_hr_array_add!(b_strian,vecdata)
  end

  diagnostic_interpolate!(b,rhs)
end

function diagnostic_residual!(
  b::DiagnosticsContribution,
  op::RBOperator{O,SplitDomains,A,<:NoHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,A}

  uh = EvaluationFunction(paramcache.trial,u)
  test = get_test(op)
  v = get_fe_basis(test)

  rhs = get_rhs(op)
  res = get_res(op)
  dc = res(r,uh,v)
  assem = get_param_assembler(op,r)

  for strian in get_domains(rhs)
    vecdata = collect_cell_vector_for_trian(test,dc,strian)
    assemble_vector_add!(b.fecache[strian],assem,vecdata)
    galerkin_projection!(b.coeff[strian],test,b.fecache[strian])
  end

  diagnostic_interpolate!(b,rhs)
end

function diagnostic_residual!(
  b::DiagnosticsContribution,
  op::RBOperator{O,T,A,<:AffineHRContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  diagnostic_interpolate!(b,op.rhs)
end

function diagnostic_residual!(
  b::DiagnosticsContribution,
  op::RBOperator{O,T,A,<:RBFContribution},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,A}

  diagnostic_interpolate!(b,op.rhs,r)
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
  op::SplitReducedOperator,
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
    assemble_hr_array_add!(A_strian,matdata)
  end

  diagnostic_interpolate!(A,lhs)
end

function diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::RBOperator{O,SplitDomains,<:NoHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,B}

  uh = EvaluationFunction(paramcache.trial,u)
  trial = get_trial(op)
  du = get_trial_fe_basis(trial)
  test = get_test(op)
  v = get_fe_basis(test)

  lhs = get_lhs(op)
  jac = get_jac(op)
  dc = jac(r,uh,du,v)
  assem = get_param_assembler(op,r)

  for strian in get_domains(lhs)
    matdata = collect_cell_matrix_for_trian(trial,test,dc,strian)
    assemble_matrix_add!(A.fecache[strian],assem,matdata)
    galerkin_projection!(A.coeff[strian],test,A.fecache[strian],trial)
  end

  diagnostic_interpolate!(A,lhs)
end

function diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::RBOperator{O,T,<:AffineHRContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  diagnostic_interpolate!(A,op.lhs)
end

function diagnostic_jacobian!(
  A::DiagnosticsContribution,
  op::RBOperator{O,T,<:RBFContribution,B},
  r::Realisation,
  u::AbstractVector,
  paramcache
  ) where {O,T,B}

  diagnostic_interpolate!(A,op.lhs,r)
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

function diagnostic_interpolate!(
  b::DiagnosticsContribution,
  a::AffineContribution,
  r::AbstractRealisation
  )

  for (ât,ct,ht) in zip(
    get_contributions(a),
    get_contributions(b.coeff),
    get_contributions(b.hypred)
    )

    interpolate!(ht,ct,ât,r)
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
- For `LinearNonlinearReducedOperator`: `"lin_rhs dim"`, `"nlin_lhs dim"`, etc.

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
    rom_diagnostics(dir,rbsolver,feop,args...;label=ONLINE_LABEL,kwargs...)
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
  label=ONLINE_LABEL,
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
    err_res,err_jac = hr_error(rbsolver,rbop,res,jac,s)
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

function offline_diagnostics(op::ReducedOperator)
  (
    state=projection_diagnostics(get_trial(op)),
    rhs=hr_diagnostics(get_rhs(op)),
    lhs=hr_diagnostics(get_lhs(op)),
  )
end

function offline_diagnostics(op::LinearNonlinearReducedOperator)
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
  N,n = _projection_dims(get_reduced_subspace(r))
  (dim=n,factor=N./n)
end

function _projection_dims(a::Projection)
  (num_fe_dofs(a),num_reduced_dofs(a))
end

function _projection_dims(a::LocalProjection)
  N = num_fe_dofs(first(local_vals(a)))
  n = maximum(map(num_reduced_dofs,local_vals(a)))
  (N,n)
end

function _projection_dims(a::BlockProjection)
  N = zeros(Int,size(a))
  n = zeros(Int,size(a))
  for i in eachindex(a)
    if a.touched[i]
      Ni,ni = _projection_dims(a[i])
      N[i] = Ni
      n[i] = ni
    end
  end
  (N,n)
end

"""
    hr_diagnostics(a::HRProjection) -> NamedTuple

Returns `(dim=n,)` for a single HR projection.
"""
function hr_diagnostics(a::HRProjection)
  n = num_reduced_dofs(a)
  (dim=n,)
end

function hr_diagnostics(a::LocalHRProjection)
  map(hr_diagnostics,local_vals(a))
end

function hr_diagnostics(a::BlockHRProjection{N}) where N
  s = size(a)
  array = Array{Any,N}(undef,s)
  for i in eachindex(a)
    if a.touched[i]
      array[i] = hr_diagnostics(a.array[i])
    end
  end
  ArrayBlock(array,a.touched)
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
  op::ReducedOperator,
  s::AbstractSnapshots
  )

  μ = get_realisation(s)
  feop = get_fe_operator(op)
  trial = get_trial(op)(μ)
  x = get_param_data(s)
  x̂ = project(trial,x)
  x̂ = inv_project(trial,x̂)
  i = get_dof_map(trial)
  ŝ = Snapshots(x̂,i,μ)
  compute_relative_error(solver,feop,s,ŝ)
end

function projection_error(
  solver::LocalRBSolver,
  op::ReducedOperator,
  s::AbstractSnapshots
  )

  gsolver = change_context(solver)
  trial = get_trial(op)
  k, = get_clusters(trial)
  cs = cluster(s,k)

  map(cs) do s  
    μ = get_realisation(s)
    opμ = get_local(op,first(μ))
    projection_error(gsolver,opμ,s)
  end |> mean 
end

"""
    hr_error(solver,op,res,jac,μ) -> (Tuple,Tuple)

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
function hr_error(::GlobalRBSolver,op::ReducedOperator,res,jac,s)
  μ = get_realisation(s)
  u = get_param_data(s)
  err_res = hr_error_res(op,res,μ,u)
  err_jac = hr_error_jac(op,jac,μ,u)
  return err_res,err_jac
end

function hr_error(::GlobalRBSolver,op::ReducedOperator{<:LinearParamEq},res,jac,s)
  μ = get_realisation(s)
  u = get_param_data(s)|> similar
  fill!(u,zero(eltype2(u)))
  err_res = hr_error_res(op,res,μ,u)
  err_jac = hr_error_jac(op,jac,μ,u)
  return err_res,err_jac
end

function hr_error(solver::LocalRBSolver,op::ReducedOperator,res,jac,s)
  μ = get_realisation(s)
  gsolver = change_context(solver)

  err_res = Any[]
  err_jac = Any[]
  for (i,μi) in enumerate(get_params(μ))
    opi = get_local(op,μi)
    si = select_snapshots(s,i)
    resi = select_snapshots(res,i)
    jaci = select_snapshots(jac,i)
    err_res_i,err_jac_i = hr_error(gsolver,opi,resi,jaci,si)
    push!(err_res,err_res_i)
    push!(err_jac,err_jac_i)
  end

  return _diag_mean(err_res),_diag_mean(err_jac)
end

for T in (:GlobalRBSolver, :LocalRBSolver)
  @eval function hr_error(solver::$T,op::LinearNonlinearReducedOperator,res,jac,s)
    res_lin,res_nlin = res
    jac_lin,jac_nlin = jac
    op_lin = get_linear_operator(op)
    op_nlin = get_nonlinear_operator(op)
    (err_res_lin,err_jac_lin) = hr_error(solver,op_lin,res_lin,jac_lin,s)
    (err_res_nlin,err_jac_nlin) = hr_error(solver,op_nlin,res_nlin,jac_nlin,s)
    return (err_res_lin,err_res_nlin),(err_jac_lin,err_jac_nlin)
  end
end 

function hr_error_res(
  op::ReducedOperator,
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
  op::ReducedOperator,
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
  fecache::AbstractParamArray,
  hypred::AbstractParamVector
  )
  
  check_interpolation(res,a,fecache)

  b̂ = get_basis(galerkin_projection(test,res))
  hrb̂ = get_all_data(hypred)

  _mean_err(b̂,hrb̂)
end

function hr_error_jac(
  trial::SingleFieldRBSpace,
  test::SingleFieldRBSpace,
  jac::Snapshots,
  a::HRProjection,
  fecache::AbstractParamArray,
  hypred::AbstractParamMatrix
  )
  
  check_interpolation(jac,a,fecache)

  μ = get_realisation(jac)
  Â = get_basis(galerkin_projection(test,jac,trial))
  Â = reshape(permutedims(Â,(1,3,2)),:,num_params(μ))
  hrÂ = reshape(get_all_data(hypred),:,num_params(μ))

  _mean_err(Â,hrÂ)
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


function hr_error_res(
  test::MultiFieldRBSpace,
  res::BlockSnapshots,
  a::BlockHRProjection,
  fecache::BlockParamVector,
  hypred::BlockParamVector
  )
  
  error = zeros(size(res))
  for i in eachindex(res)
    if res.touched[i]
      error[i] = hr_error_res(test[i],res[i],a[i],fecache.data[i],hypred.data[i])
    end
  end
  error
end

function hr_error_jac(
  trial::MultiFieldRBSpace,
  test::MultiFieldRBSpace,
  jac::BlockSnapshots,
  a::BlockHRProjection,
  fecache::BlockParamMatrix,
  hypred::BlockParamMatrix
  )
  
  error = zeros(size(jac))
  for i in axes(jac,1), j in axes(jac,2)
    if jac.touched[i,j]
      error[i,j] = hr_error_jac(trial[j],test[i],jac[i,j],a[i,j],fecache.data[i,j],hypred.data[i,j])
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
  save(dir,res;label=_get_label(label,RESIDUALS_LABEL))
end

function save_jacobians(dir,feop,jac;label="")
  save(dir,jac;label=_get_label(label,JACOBIANS_LABEL))
end

for f in (:save_residuals,:save_jacobians)
  @eval begin
    function $f(dir,feop::LinearNonlinearParamOperator,resjac::Tuple;label="")
      @assert length(resjac) == 2
      $f(dir,get_linear_operator(feop),resjac[1];label=_get_label(label,LINEAR_LABEL))
      $f(dir,get_nonlinear_operator(feop),resjac[2];label=_get_label(label,NONLINEAR_LABEL))
      return
    end
  end
end

function load_residuals(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_res(feop);label=_get_label(label,RESIDUALS_LABEL))
end

function load_jacobians(dir,feop::ParamOperator;label="")
  load_contribution(dir,get_domains_jac(feop);label=_get_label(label,JACOBIANS_LABEL))
end

for f in (:load_residuals,:load_jacobians)
  @eval begin
    function $f(dir,feop::LinearNonlinearParamOperator;label="")
      (
        $f(dir,get_linear_operator(feop);label=_get_label(label,LINEAR_LABEL)),
        $f(dir,get_nonlinear_operator(feop);label=_get_label(label,NONLINEAR_LABEL)),
      )
    end
  end
end

function load_residuals(dir,rbsolver,feop,fesnaps;label=ONLINE_LABEL)
  try
    res = load_residuals(dir,feop;label)
    select_snapshots(res,res_params(rbsolver))
  catch
    res = residual_snapshots(rbsolver,feop,fesnaps)
    save_residuals(dir,feop,res;label)
    res
  end
end

function load_jacobians(dir,rbsolver,feop,fesnaps;label=ONLINE_LABEL)
  try
    jac = load_jacobians(dir,feop;label)
    select_snapshots(jac,jac_params(rbsolver))
  catch
    jac = jacobian_snapshots(rbsolver,feop,fesnaps)
    save_jacobians(dir,feop,jac;label)
    jac
  end
end

function load_problem_snapshots(
  dir,rbsolver,feop,args...;
  nparams=:all,label=ONLINE_LABEL,kwargs...
  )

  s = load_snapshots(dir,rbsolver,feop,args...;label,kwargs...)
  s = nparams == :all ? s : select_snapshots(s,1:nparams)
  try
    rbsolver = set_params(rbsolver;nparams=num_params(s))
  catch
    s = select_snapshots(s,1:1)
  end
  jac = load_jacobians(dir,rbsolver,feop,s;label)
  res = load_residuals(dir,rbsolver,feop,s;label)
  return s,jac,res
end

# utils

function check_interpolation(res,a::HRVecProjection,fecache)
  msg = "fecache mismatch at interpolation points"
  rows = get_interpolation_dofs(get_interpolation(a))
  bdata = flatten(res)
  @check isapprox(get_all_data(fecache),bdata[rows,:];rtol=1e-8) msg
  return true
end

function check_interpolation(jac,a::HRMatProjection,fecache)
  msg = "fecache mismatch at interpolation points"
  rows,cols = get_interpolation_dofs(get_interpolation(a))
  sparsity = get_sparsity(get_dof_map(jac))
  inds = sparsify_split_indices(rows,cols,sparsity)
  Adata = flatten(jac)
  @check isapprox(get_all_data(fecache),Adata[inds,:];rtol=1e-8) msg
  return true
end

for S in (:HRVecProjection,:HRMatProjection), T in (:RBFHyperReduction,:TrivialHyperReduction)
  @eval function check_interpolation(resjac,a::$S{<:$T},fecache)
    return true
  end
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
  SupremizerReduction(set_params(red.reduction;kwargs...),red.coupling,red.supr_tol)
end

function set_params(red::TrivialHyperReduction;nparams::Int)
  nparams == 1 && return red
  @notimplemented "Cannot set parameters for TrivialHyperReduction"
end

function set_params(red::DEIMHyperReduction;kwargs...)
  DEIMHyperReduction(set_params(red.reduction;kwargs...))
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

function _mean_err(a::AbstractMatrix,â::AbstractMatrix)
  @check size(a) == size(â) 
  err = 0.0
  ε = eps()
  @inbounds @views for i in axes(a,2)
    err += norm(a[:,i]-â[:,i]) / (norm(a[:,i]) + ε)
  end
  err / size(a,2)
end

function _entries_to_dict(entries::Vector{<:NamedTuple})
  d = Dict{String,Any}()
  isempty(entries) && return d
  d["tols"] = [e.tol for e in entries]
  _unpack_into_dict!(d,"",map(e -> e.diagnostics,entries))
  return d
end

_diag_mean(vals::AbstractVector{<:Number}) = mean(vals)

function _diag_mean(vals::AbstractVector{<:Tuple})
  n = length(first(vals))
  ntuple(i -> _diag_mean(map(v -> v[i],vals)),n)
end

function _diag_mean(vals::AbstractVector{<:AbstractArray})
  v0 = first(vals)
  if eltype(v0) <: Number
    out = similar(v0,Float64)
    for i in eachindex(v0)
      out[i] = _diag_mean(map(v -> v[i],vals))
    end
    return out
  end

  out = Array{Any}(undef,size(v0))
  for i in eachindex(v0)
    out[i] = _diag_mean(map(v -> v[i],vals))
  end
  out
end

function _diag_mean(vals::AbstractVector)
  v0 = first(vals)
  if v0 isa Number
    return mean(vals)
  elseif v0 isa Tuple
    n = length(v0)
    return ntuple(i -> _diag_mean([v[i] for v in vals]),n)
  elseif v0 isa AbstractArray
    if eltype(v0) <: Number
      out = similar(v0,Float64)
      for i in eachindex(v0)
        out[i] = _diag_mean([v[i] for v in vals])
      end
      return out
    end

    out = Array{Any}(undef,size(v0))
    for i in eachindex(v0)
      out[i] = _diag_mean([v[i] for v in vals])
    end
    return out
  end

  @notimplemented
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