module PoissonEmbedded

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main_2d(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4)
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  R = 0.3
  pmin = Point(0,0)
  pmax = Point(1,1)
  dp = pmax - pmin

  geo = ! disk(R,x0=Point(0.5,0.5))

  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(pmin,pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
  end

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)
  Γn = BoundaryTriangulation(bgmodel,tags=[8])

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)
  dΓn = Measure(Γn,degree)

  n_Γ = get_normal_vector(Γ)
  nΓ = get_normal_vector(Γ)

  γd = 10.0
  hd = dp[1]/n

  ν(μ) = x->μ[3]
  νμ(μ) = ParamFunction(ν,μ)

  f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
  fμ(μ) = ParamFunction(f,μ)

  h(μ) = x->1
  hμ(μ) = ParamFunction(h,μ)

  g(μ) = x->μ[3]*x[1]-x[2]
  gμ(μ) = ParamFunction(g,μ)

  a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
  res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

  trian_a = (Ω,Γ)
  trian_res = (Ω,Γ,Γn)
  domains = FEDomains(trian_res,trian_a)

  reffe = ReferenceFE(lagrangian,Float64,order)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg = FESpace(Ωbg,reffe,conformity=:H1)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = ParamTrialFESpace(test,gμ)
  feop = ExtensionLinearParamOperator(res,a,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
  tolrank = tol_or_rank(tol,rank)
  tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
  state_reduction = Reduction(tolrank,energy;nparams)

  fesolver = ExtensionSolver(LUSolver())
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

function main_3d(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4)
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  geo = popcorn()
  box = get_metadata(geo)
  partition = (n,n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(box.pmin,box.pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(box.pmin,box.pmax,partition)
  end

  order = 2
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

  ν(μ) = x->μ[3]
  νμ(μ) = ParamFunction(ν,μ)

  f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x->μ[3]*x[1]-x[2]
  gμ(μ) = ParamFunction(g,μ)

  a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(μ,v,dΩ,dΓ) = ∫(fμ(μ)⋅v)dΩ + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
  res(μ,u,v,dΩ,dΓ) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ)

  trian_a = (Ω,Γ)
  trian_res = (Ω,Γ)
  domains = FEDomains(trian_res,trian_a)

  reffe = ReferenceFE(lagrangian,Float64,order)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg = FESpace(Ωbg,reffe,conformity=:H1)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = ParamTrialFESpace(test,gμ)
  feop = ExtensionLinearParamOperator(res,a,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
  tolrank = tol_or_rank(tol,rank)
  tolrank = method == :ttsvd ? fill(tolrank,3) : tolrank
  state_reduction = Reduction(tolrank,energy;nparams)

  fesolver = ExtensionSolver(LUSolver())
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

# main_2d(:pod)
# main_2d(:ttsvd)
# main_3d(:pod)
main_3d(:ttsvd)

end



using Gridap
using GridapEmbedded
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.Extensions

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=20
tol=1e-4
rank=nothing
nparams=50
nparams_res=nparams
nparams_jac=nparams
@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
dp = pmax - pmin

geo = ! disk(R,x0=Point(0.5,0.5))

partition = (n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

order = 2
degree = 2*order

cutgeo = cut(bgmodel,geo)
Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(bgmodel,tags=[8])

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓn = Measure(Γn,degree)

n_Γ = get_normal_vector(Γ)
nΓ = get_normal_vector(Γ)

γd = 10.0
hd = dp[1]/n

trian_a = (Ω,Γ)
trian_res = (Ω,Γ)
domains = FEDomains(trian_res,trian_a)

reffe = ReferenceFE(lagrangian,Float64,order)

# agfem
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)
testbg = FESpace(Ωbg,reffe,conformity=:H1)
testact = FESpace(Ωact,reffe,conformity=:H1)
testagg = AgFEMSpace(testact,aggregates)

test = DirectSumFESpace(testbg,testagg)

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
ncentroids = 10
state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

function def_fe_operator(μ)
  ν = x->μ[3]
  f = x->μ[1]*x[1] - μ[2]*x[2]
  h = x->1
  g = x->μ[3]*x[1]-x[2]

  a(u,v,dΩ,dΓ) = ∫(ν*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v,dΩ,dΓ) =  a(u,v,dΩ,dΓ) - l(v,dΩ,dΓ)

  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

μ = realization(pspace;nparams)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

# offline
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

# online
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

# test
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
println(perf)

μ = μon.params[1]
opμ = RBSteady.get_local(rbop,μ)
U = get_trial(opμ)(nothing)
x̂ = zero_free_values(U)
syscache = allocate_systemcache(opμ,x̂)
solve!(x̂,fesolver,opμ,syscache)

xvec = map(x̂,μon) do x̂,μ
  opμ = RBSteady.get_local(rbop,μ)
  trial = RBSteady.get_trial(opμ)
  inv_project(trial,x̂)
end

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

μtest = realization(feop;sampling=:uniform)
stest, = solution_snapshots(rbsolver,feop,μtest)
rtest = residual_snapshots(rbsolver1,feop,stest)
jtest = jacobian_snapshots(rbsolver1,feop,stest)
k = RBSteady.get_clusters(rbop.test)
lab = RBSteady.get_label(k,μtest.params[1])
resvec = RBSteady.cluster_snapshots(ress,k)
jacvec = RBSteady.cluster_snapshots(jacs,k)

s,V = resvec[lab],RBSteady.local_values(rbop.test)[lab]
hr = HRProjection(rbsolver.residual_reduction.reduction,s,V)
proj = projection(rbsolver.residual_reduction.reduction.reduction,s)

coeff = interpolate(hr.interpolation,μtest)
r̂test = get_basis(proj)*coeff[1]

maximum(abs.(rtest - r̂test))

s,U = jacvec[lab],RBSteady.local_values(rbop.trial)[lab]
hr = HRProjection(rbsolver.jacobian_reduction.reduction,s,U,V)
proj = projection(rbsolver.jacobian_reduction.reduction.reduction,s)
Φ = proj.basis.data

coeff = interpolate(hr.interpolation,μtest)
ĵtest = sum(map(i -> get_basis(proj)[i,i]*coeff[1][i],eachindex(coeff[1])))
Jtest = recast(jtest,jtest.dof_map.sparsity)[1]
maximum(abs.(Jtest - ĵtest))
