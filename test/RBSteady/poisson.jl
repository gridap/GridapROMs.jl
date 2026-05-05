module PoissonEquation

using Gridap
using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt) ? hypred_strategy : :mdeim

  println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1]/sum(μ))
  aμ(μ) = parameterise(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = parameterise(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = parameterise(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = parameterise(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  if method == :pod
    state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:sopt)
  main(method,compression,hypred_strategy)
end

end

using Gridap
using GridapROMs

method=:pod
compression=:global
hypred_strategy=:none
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

  println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (20,20)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1]/sum(μ))
  aμ(μ) = parameterise(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = parameterise(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = parameterise(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = parameterise(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  if method == :pod
    state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=8,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  println(perf)

  using Gridap.Algebra
  using Gridap.FESpaces
  using GridapROMs.ParamSteady 
  using GridapROMs.ParamAlgebra
  using GridapROMs.ParamDataStructures

  nlop = parameterise(rbop,μon)
  x̂ = zero_free_values(rbop.trial(μon))
  A = allocate_jacobian(nlop,x̂)
  jacobian!(A,nlop,x̂)
  b = allocate_residual(nlop,x̂)
  residual!(b,nlop,x̂)

  fe_nlop = parameterise(feop,μon)
  feA = allocate_jacobian(fe_nlop,x̂.fe_data)
  jacobian!(feA,fe_nlop.op,fe_nlop.μ,x̂.fe_data,fe_nlop.paramcache)
  feb = allocate_residual(fe_nlop,x̂.fe_data)
  residual!(feb,fe_nlop.op,fe_nlop.μ,x̂.fe_data,fe_nlop.paramcache)

  x̂_man = project(rbop.test,get_param_data(x))
  A_man = sum(feA.values)
  b_man = sum(feb.values)
  x̂_manman = ParamArray(map(i -> A_man[i,i] \ b_man[i],1:10))
  Φ = get_basis(rbop.test)
  Â_man1 = Φ' * A_man[5,5] * Φ
  b̂_man1 = Φ' * b_man[1]

  Â_man1 ≈ sum(A.coeff.values).data[:,5,:]
  b̂_man1 ≈ sum(b.coeff.values).data[1]

  A_man[1,1] ≈ sum(A.fecache.values)[1,1]

  galerkin_projection!(A.coeff[1],Φ,A.fecache[1],Φ)
  # U = get_trial(rbop)(μon)
  # x̂ = zero_free_values(U)

  # nlop = parameterise(rbop,μon)
  # syscache = allocate_systemcache(nlop,x̂)

  # # solve!(x̂,fesolver,nlop,syscache)
  # fill!(x̂,zero(eltype(x̂)))
  
  # residual!(syscache.b,nlop,x̂)
  # jacobian!(syscache.A,nlop,x̂)
  # ye1 = syscache.A[1,1] \ syscache.b[1]