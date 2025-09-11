module BurgersEquation

using Gridap
using GridapROMs

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt) ? hypred_strategy : :mdeim

  println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

  pdomain = (1,10,1,10)

  domain = (-1,1)
  partition = (128,)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 3*order-1

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  u(μ,t) = x -> 5 + 2sin(μ[1]*x[1]+μ[2]*t)
  uμt(μ,t) = parameterize(u,μ,t)

  f(μ,t) = x -> 2*μ[2]*cos(μ[1]*x[1]+μ[2]*t) + u(μ,t)(x[1])*2*μ[1]*cos(μ[1]*x[1]+μ[2]*t)
  fμt(μ,t) = parameterize(f,μ,t)

  stiffness(μ,t,u,v,dΩ) = ∫(0.0*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
  res(μ,t,u,v,dΩ) = ∫(v⋅∂t(u))dΩ - ∫(v*fμt(μ,t))dΩ

  res_nlin(μ,t,u,v,dΩ) = 0.5*∫( ∇(v)⋅VectorValue(1,)*u*u )dΩ
  jac_nlin(μ,t,u,du,v,dΩ) = ∫( ∇(v)⋅VectorValue(1)*u*du )dΩ

  trian_res = (Ω,)
  trian_jac = (Ω,)
  trian_jac_t = (Ω,)
  domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
  domains_nlin = FEDomains(trian_res,(trian_jac,))

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;dirichlet_tags="boundary")
  trial = TransientTrialParamFESpace(test,uμt)

  θ = 0.5
  dt = 0.005
  t0 = 0.0
  tf = 10*dt
  tdomain = t0:dt:tf

  ptspace = TransientParamSpace(pdomain,tdomain)

  u0(μ) = u(μ,t0)
  u0μ(μ) = parameterize(u0,μ)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  if method == :pod
    state_reduction = HighDimReduction(tol,energy;nparams,sketch,compression,ncentroids)
  else method == :ttsvd
    state_reduction = HighDimReduction(fill(tol,2),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop_lin = TransientLinearParamOperator(res,(stiffness,mass),ptspace,
    trial,test,domains_lin)
  feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
    trial,test,domains_nlin)
  feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

  fesnaps,= solution_snapshots(rbsolver,feop,uh0μ)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
  x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd),compression in (:local,:global),hypred_strategy in (:mdeim,)
  main(method,compression,hypred_strategy)
end

end



using Gridap
using GridapROMs

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

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
hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt) ? hypred_strategy : :mdeim

println("Running test with compression $method, $compression compressions, and $hypred_strategy hyper-reduction")

my_sign(Fn) = Fn[1] < 0.0 ? -0.5 : 0.5

pdomain = (4.25,5.5,0.015,0.03)

const n = 256
const h = 1/n
const CFL = 0.1

domain = (0,100)
partition = (n,)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 1
degree = 3*(order+1)

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

Λ = SkeletonTriangulation(model)
dΛ = Measure(Λ,degree)
n_Λ = get_normal_vector(Λ)

Γ = BoundaryTriangulation(model,tags=[1])
dΓ = Measure(Γ,degree)
n_Γ = get_normal_vector(Γ)

f(μ,t) = x -> 0.02*exp(μ[2]*x[1])
fμt(μ,t) = parameterize(f,μ,t)

g(μ,t) = x -> μ[1]
gμt(μ,t) = parameterize(g,μ,t)

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:L2)
trial = TransientTrialParamFESpace(test)

# # project velocity onto Hdiv
# Vdiv = TestFESpace(model,ReferenceFE(raviart_thomas,Float64,order),conformity=:Hdiv)
# Udiv = TrialFESpace(Vdiv)
# adiv(u,v) = ∫(u*v)dΩ
# ldiv(v) = ∫(0.5*u*v)dΩ
# op = AffineFEOperator(adiv,ldiv,U,V)
# F = solve(op)
F(u) = 0.5*VectorValue.(u)

astab(u,v,dΛ) = ∫( mean(F(u)*u)⋅jump(v*n_Λ) + (my_sign∘((F(u)⋅n_Λ).plus)*((F(u)⋅n_Λ).plus))*jump(u*n_Λ)⋅jump(v*n_Λ) )dΛ
lstab(μ,t,v,dΓ) = ∫( (F(u)⋅n_Γ)⋅gμt(μ,t)*v )dΓ

stiffness(μ,t,u,v,dΛ) = astab(u,v,dΛ)
mass(μ,t,uₜ,v,dΩ) = ∫(v⋅uₜ)dΩ
res(μ,t,u,v,dΩ,dΓ) = ∫(v⋅∂t(u))dΩ - ∫(v*fμt(μ,t))dΩ - lstab(μ,t,v,dΓ)

res_nlin(μ,t,u,v,dΩ) = ∫(-u*(∇(v)⋅F(u)))dΩ
jac_nlin(μ,t,u,du,v,dΩ) = ∫(2*(∇(v)⋅F(u))*du)dΩ

domains_lin = FEDomains((Ω,Γ),((Λ,),(Ω,)))
domains_nlin = FEDomains((Ω,),((Ω,),))

energy(u,v) = ∫(∇(v)⋅∇(u))dΩ

θ = 1.0
dt = (CFL*1/n)/order
t0 = 0.0
tf = 50*dt
tdomain = t0:dt:tf

ptspace = TransientParamSpace(pdomain,tdomain)

u0(μ) = x -> 1.0
u0μ(μ) = parameterize(u0,μ)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

if method == :pod
  state_reduction = HighDimReduction(tol,energy;nparams,sketch,compression,ncentroids)
else method == :ttsvd
  state_reduction = HighDimReduction(fill(tol,2),energy;nparams,sketch,compression,ncentroids)
end

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

feop_lin = TransientLinearParamOperator(res,(stiffness,mass),ptspace,
  trial,test,domains_lin)
feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
  trial,test,domains_nlin)
feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)

fesnaps,= solution_snapshots(rbsolver,feop,uh0μ)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
x,festats = solution_snapshots(rbsolver,feop,μon,uh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

println(perf)

# jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
# reduced_operator(rbsolver,feop,fesnaps)

# oplin = rbop.op_linear
# ϕs = get_basis(oplin.test.subspace.projection_space)
# ϕt = get_basis(oplin.test.subspace.projection_time)
# ϕ = get_basis(oplin.test.subspace)
# X = assemble_matrix(feop,energy)
# ϕs'X*ϕs
# ϕt'ϕt
# xon = reshape(x,:,10)
# xon - inv_project(oplin.test.subspace,project(oplin.test.subspace,xon))

# a = oplin.test.subspace
# x̂ = allocate_in_domain(a,xon)
# project!(x̂,a,xon)

# S = project(oplin.test.subspace,get_param_data(x))
# iS = inv_project(oplin.test.subspace,S)
# y = get_param_data(x)
# x̂ = allocate_in_domain(a,y)
# project!(x̂,a,y)
