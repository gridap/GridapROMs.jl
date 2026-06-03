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
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (10,10)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
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
  perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:sopt,:rbf,:none,:affine)
  main(method,compression,hypred_strategy)
end

end


using Gridap
using GridapROMs
using GridapROMs.RBSteady

method=:pod
compression=:global
hypred_strategy=:mdeim
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

println("Running test with $compression ($method, $hypred_strategy) strategy")

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (10,10)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
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

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
du,v = get_trial_fe_basis(red_trial),get_fe_basis(red_test)
μ = realisation(feop;nparams=10) 
dc = ∫(aμ(μ)*∇(v)⋅∇(du))dΩ
trian = Ω

using GridapROMs.RBSteady 
using Gridap.Arrays
using Gridap.Geometry
using Gridap.ReferenceFEs
using Gridap.CellData

ℛstiffness(μ,u,v,dΩ) = ∫(ℛ(aμ(μ);order)*∇(v)⋅∇(u))dΩ
ℛrhs(μ,v,dΩ,dΓn) = ∫(ℛ(fμ(μ);order)*v)dΩ + ∫(ℛ(hμ(μ);order)*v)dΓn
ℛres(μ,u,v,dΩ,dΓn) = ℛstiffness(μ,u,v,dΩ) - ℛrhs(μ,v,dΩ,dΓn)

ℛfeop = LinearParamOperator(ℛres,ℛstiffness,pspace,trial,test,domains)
jacs = jacobian_snapshots(rbsolver,ℛfeop,fesnaps)
ress = residual_snapshots(rbsolver,ℛfeop,fesnaps)

cf = aμ(μ)*∇(v)⋅∇(uh)
_cf = ℛ(aμ(μ);order)*∇(v)⋅∇(uh)

# cfx = cf.args[1].args[1](x)
# _cfx = _cf.args[1].args[1](x)

ax = map(i->i(x),cf.args)
eltype(ax[1])
eltype(ax[2])
k = Fields.BroadcastingFieldOpMap(⋅)
# lazy_map(k,ax...)
c = return_cache(k,ax[1][1],ax[2][1]) 
evaluate!(c,k,ax[1][1],ax[2][1])

_ax = map(i->i(x),_cf.args)
eltype(_ax[1])
eltype(_ax[2])
# lazy_map(k,_ax...)
_c = return_cache(k,_ax[1][1],_ax[2][1]) 
evaluate!(_c,k,_ax[1][1],_ax[2][1])

idk = (ℛ(aμ(μ);order)*∇(v)⋅∇(du))(x)
cache = array_cache(idk)
getindex!(cache,idk,10)

cvm = (∫(aμ(μ)*∇(v)⋅∇(du))dΩ)[Ω]
cvv = (∫(aμ(μ)*v)dΩ)[Ω]