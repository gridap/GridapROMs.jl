module HeatEquationEmbedded

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,1,10,1,10)

  geo = popcorn()
  box = get_metadata(geo)
  partition = (n,n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(box.pmin,box.pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(box.pmin,box.pmax,partition)
  end

  order = 1
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)
  nΓ = get_normal_vector(Γ)

  γd = 10.0
  dp = box.pmax - box.pmin
  hd = dp[1]/n

  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf
  pdomain = (1,10,1,10,1,10)
  ptspace = TransientParamSpace(pdomain,tdomain)

  a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
  aμt(μ,t) = TransientParamFunction(a,μ,t)

  f(μ,t) = x -> 1.
  fμt(μ,t) = TransientParamFunction(f,μ,t)

  h(μ,t) = x -> abs(cos(t/μ[3]))
  hμt(μ,t) = TransientParamFunction(h,μ,t)

  g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  u0(μ) = x -> 0.0
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ,dΓ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓ) = ∫(fμt(μ,t)*v)dΩ + ∫( (γd/hd)*v*gμt(μ,t) - (n_Γ⋅∇(v))*gμt(μ,t) )dΓ
  res(μ,t,u,v,dΩ,dΓ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ,dΓ) - rhs(μ,t,v,dΩ,dΓ)

  trian_res = (Ω,Γ)
  trian_stiffness = (Ω,Γ)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg

  reffe = ReferenceFE(lagrangian,Float64,order)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg = FESpace(Ωbg,reffe,conformity=:H1)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientExtensionLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)

  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
  tolrank = tol_or_rank(tol,rank)
  tolrank = method == :ttsvd ? fill(tolrank,4) : tolrank
  state_reduction = TransientReduction(tolrank,energy;nparams)

  fesolver = ThetaMethod(ExtensionSolver(LUSolver()),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)

  x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

# main(:pod)
main(:ttsvd)

end

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:ttsvd
n=20
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
nparams_djac=1
sketch=:sprn

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
pdomain = (1,10,1,10,1,10)

geo = popcorn()
box = get_metadata(geo)
partition = (n,n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(box.pmin,box.pmax,partition)
else
  bgmodel = CartesianDiscreteModel(box.pmin,box.pmax,partition)
end

order = 1
degree = 2*order

cutgeo = cut(bgmodel,geo)
Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

n_Γ = get_normal_vector(Γ)
nΓ = get_normal_vector(Γ)

γd = 10.0
dp = box.pmax - box.pmin
hd = dp[1]/n

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf
pdomain = (1,10,1,10,1,10)
ptspace = TransientParamSpace(pdomain,tdomain)

a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
aμt(μ,t) = TransientParamFunction(a,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[3]))
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ,dΓ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓ) = ∫(fμt(μ,t)*v)dΩ + ∫( (γd/hd)*v*gμt(μ,t) - (n_Γ⋅∇(v))*gμt(μ,t) )dΓ
res(μ,t,u,v,dΩ,dΓ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ,dΓ) - rhs(μ,t,v,dΩ,dΓ)

trian_res = (Ω,Γ)
trian_stiffness = (Ω,Γ)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg

reffe = ReferenceFE(lagrangian,Float64,order)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)
testbg = FESpace(Ωbg,reffe,conformity=:H1)
testact = FESpace(Ωact,reffe,conformity=:H1)
testagg = AgFEMSpace(testact,aggregates)

test = DirectSumFESpace(testbg,testagg)
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientExtensionLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,4) : tolrank
state_reduction = TransientReduction(tolrank,energy;nparams)

fesolver = ThetaMethod(ExtensionSolver(LUSolver()),dt,θ)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,sampling=:uniform)
# x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
using Gridap.FESpaces
using Gridap.Algebra
using GridapROMs.ParamAlgebra
r,op = μon,rbop
U = get_trial(op)(r)
x̂ = zero_free_values(U)
nlop = parameterize(op,r)
syscache = allocate_systemcache(nlop,x̂)
A,b = syscache.A,syscache.b
jacobian!(A,nlop,x̂)
residual!(b,nlop,x̂)

# x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
# perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

using GridapROMs.RBTransient
using GridapROMs.RBSteady

s = jacs[1][1]
red = rbsolver.residual_reduction.reduction
basis = projection(red,s)

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
proj_left = red_test.subspace
proj_right = proj_left
proj = basis

p_space = galerkin_projection(
  RBTransient.get_basis_space(proj_left),
  RBTransient.get_basis_space(proj),
  RBTransient.get_basis_space(proj_right))

p_time = galerkin_projection(
  RBTransient.get_core_time(proj_left),
  RBTransient.get_basis_time(proj),
  RBTransient.get_core_time(proj_right),
  combine)

# pod interface

data = reshape(fesnaps,:,size(fesnaps,4),size(fesnaps,5))
A = change_snaps_dof_map(fesnaps,dof_map)
Xk = kron(assemble_matrix(feop,energy))
kred = TransientReduction(1e-4,energy;nparams)
Φ = RBTransient.KroneckerProjection(kred,A,Xk)

_p_space = galerkin_projection(
  RBTransient.get_basis_space(Φ),
  RBTransient.get_basis_space(proj),
  RBTransient.get_basis_space(Φ))


#
ffesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rrrbop = reduced_operator(rbsolver,feop,ffesnaps)
x̂x̂,rbstats = solve(rbsolver,rrrbop,μon,uh0μ)

jjacs = jacobian_snapshots(rbsolver,feop,fesnaps)

ss = jjacs[1][1]
basis = projection(kred,ss)

rred_trial,rred_test = reduced_spaces(rbsolver,feop,fesnaps)
proj_left = rred_test.subspace
proj_right = proj_left
proj = basis

p_space = galerkin_projection(
  RBTransient.get_basis_space(proj_left),
  RBTransient.get_basis_space(proj),
  RBTransient.get_basis_space(proj_right))

p_time = galerkin_projection(
  RBTransient.get_core_time(proj_left),
  RBTransient.get_basis_time(proj),
  RBTransient.get_core_time(proj_right),
  combine)
