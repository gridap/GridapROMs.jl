module PoissonEquation

using Gridap
using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=4
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:rbf) ? hypred_strategy : :mdeim

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
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = ParamFunction(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = ParamFunction(h,μ)

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
  else method == :ttsvd
    state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:rbf)
  main(method,compression,hypred_strategy)
end

end



using Gridap
using GridapROMs

method=:ttsvd
compression=:local
hypred_strategy=:mdeim
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:mdeim,:rbf) ? hypred_strategy : :mdeim

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
aμ(μ) = ParamFunction(a,μ)

f(μ) = x -> 1.
fμ(μ) = ParamFunction(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = ParamFunction(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = ParamFunction(h,μ)

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
  state_reduction = Reduction(tol,energy;nparams,sketch,compression)
else method == :ttsvd
  state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression)
end

fesolver = LUSolver()
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
feop_uniform = LinearParamOperator(res,stiffness,pspace_uniform,trial,test,domains)
μon = realization(feop_uniform;nparams=10)
x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

pspace = ParamSpace(pdomain)
feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)
x̂,rbstats = solve(rbsolver,rbop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using GridapROMs.Utils
using GridapROMs.RBSteady
using GridapROMs.ParamAlgebra
using Gridap.FESpaces
using Gridap.Algebra
using Gridap.ODEs

opμ = get_local(rbop,first(μon))
# x̂,stats = solve(rbsolver,opμ,μon[1])
U = get_trial(opμ)(μon[1])
x̂ = zero_free_values(U)

nlop = parameterize(opμ,μon[1])
syscache = allocate_systemcache(nlop,x̂)
b = syscache.b
# residual!(b,nlop,x̂)
fill!(b,zero(eltype(b)))
uh = EvaluationFunction(nlop.paramcache.trial,x̂)
v = get_fe_basis(test)
trian_res = get_domains_res(opμ.op)
dc = get_res(opμ.op)(μon[1],uh,v)
strian = trian_res[1]
b_strian = b.fecache[strian]
rhs_strian = get_interpolation(opμ.rhs[strian])
vecdata = collect_cell_hr_vector(test,dc,strian,rhs_strian)
# assemble_hr_vector_add!(b_strian,vecdata...)
cellvec,cellidsrows,icells = vecdata
rows_cache = array_cache(cellidsrows)
vals_cache = array_cache(cellvec)
vals1 = getindex!(vals_cache,cellvec,1)
rows1 = getindex!(rows_cache,cellidsrows,1)
add! = RBSteady.AddHREntriesMap(+)
add_cache = return_cache(add!,b,vals1,rows1)
caches = add!,add_cache,vals_cache,rows_cache
