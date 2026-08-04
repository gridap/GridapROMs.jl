module RingFisherKPP

using Gridap
using GridapGmsh
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using DrWatson

using GridapROMs

# Fisher-Kolmogorov-Petrovski-Piskunov reaction-diffusion equation on a quarter
# annulus Ω = {(ρ,ϑ) ∈ (1,1.5) × (0,π/2)}:
#
#   ∂u/∂t - div(ν(x;μ)∇u) + N(u) = 0,  N(u) = 75*u*(1-u)
#   ν(x;μ)∇u·n = 0 on ∂Ω  (homogeneous Neumann, so no Dirichlet BCs are needed)
#   u(0;μ) = exp(-((x1-1.5)^2 + 50*x2^2))
#
# ν(x;μ) = μ[1] is taken as a constant (in space), scalar diffusivity.
function main(
  compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=20,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running Fisher-KPP test with $compression ($hypred_strategy) strategy")

  pdomain = (0.01,1.0)

  model = GmshDiscreteModel(datadir("models/quarter_annulus.msh"))

  order = 1
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  ν(μ) = x -> μ[1]
  νμ(μ) = parameterise(ν,μ)

  γ = 75.0

  u0(μ) = x -> exp(-((x[1]-1.5)^2+50*x[2]^2))
  u0μ(μ) = parameterise(u0,μ)

  # linear part: mass + diffusion + the linear (75*u) piece of the reaction term
  stiffness(μ,t,u,v,dΩ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫(γ*v*u)dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  res(μ,t,u,v,dΩ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ)

  # nonlinear part: the remaining -75*u^2 piece of the reaction term
  res_nlin(μ,t,u,v,dΩ) = ∫(-γ*v*(u*u))dΩ
  jac_nlin(μ,t,u,du,v,dΩ) = ∫(-2*γ*v*u*du)dΩ

  trian_res = (Ω,)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains_lin = FEDomains(trian_res,(trian_stiffness,trian_mass))
  domains_nlin = FEDomains(trian_res,(trian_stiffness,))

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1)
  trial = TransientTrialParamFESpace(test)

  θ = 1.0             # implicit (backward) Euler
  dt = 1.1e-3
  t0 = 0.0
  Nt = 140
  tf = t0+Nt*dt
  tdomain = t0:dt:tf

  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  state_reduction = HighDimReduction(tol,energy;nparams,sketch,compression,ncentroids)

  ptspace = TransientParamSpace(pdomain,tdomain)

  fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)

  feop_lin = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains_lin)
  feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,trial,test,domains_nlin)
  feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

  fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
  x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
  perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  println(perf)
end

main(:global,:none)

end

using Gridap
using GridapGmsh
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using DrWatson

using GridapROMs

compression=:global
hypred_strategy=:mdeim
tol=1e-4
nparams=20
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

pdomain = (0.01,1.0)

model = GmshDiscreteModel(datadir("models/quarter_annulus.msh"))

order = 1
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

ν(μ) = x -> μ[1]
νμ(μ) = parameterise(ν,μ)

γ = 75.0

u0(μ) = x -> exp(-((x[1]-1.5)^2+50*x[2]^2))
u0μ(μ) = parameterise(u0,μ)

# linear part: mass + diffusion + the linear (75*u) piece of the reaction term
stiffness(μ,t,u,v,dΩ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫(γ*v*u)dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
res(μ,t,u,v,dΩ) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ)

# nonlinear part: the remaining -75*u^2 piece of the reaction term
res_nlin(μ,t,u,v,dΩ) = ∫(-γ*v*(u*u))dΩ
jac_nlin(μ,t,u,du,v,dΩ) = ∫(-2*γ*v*u*du)dΩ

trian_res = (Ω,)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains_lin = FEDomains(trian_res,(trian_stiffness,trian_mass))
domains_nlin = FEDomains(trian_res,(trian_stiffness,))

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1)
trial = TransientTrialParamFESpace(test)

θ = 1.0             # implicit (backward) Euler
dt = 1.1e-3
t0 = 0.0
Nt = 140
tf = t0+Nt*dt
tdomain = t0:dt:tf

uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

state_reduction = SteadyReduction(tol,energy;nparams,sketch,compression,ncentroids)

ptspace = TransientParamSpace(pdomain,tdomain)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jacs=(nparams_jac,nparams_jac),hypred_strategy)

feop_lin = TransientLinearParamOperator(res,(stiffness,mass),ptspace,trial,test,domains_lin)
feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,trial,test,domains_nlin)
feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realisation(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)