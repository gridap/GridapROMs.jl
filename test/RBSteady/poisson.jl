module PoissonEquation

using Gridap
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
  pdomain = (1,10,1,10,1,10)

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

  tolrank = tol_or_rank(tol,rank)
  if method == :pod
    state_reduction = PODReduction(tolrank,energy;nparams,sketch)
  else method == :ttsvd
    tolranks = fill(tolrank,3)
    state_reduction = Reduction(tolranks,energy;nparams)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  pspace_uniform = ParamSpace(pdomain;sampling=:uniform)
  feop_uniform = LinearParamOperator(res,stiffness,pspace_uniform,trial,test,domains)
  μon = realization(feop_uniform;nparams=10)
  x,festats = solution_snapshots(rbsolver,feop_uniform,μon)

  for sampling in (:uniform,:halton,:latin_hypercube,:tensorial_uniform)
    println("Running $method test with sampling strategy $sampling")
    pspace = ParamSpace(pdomain;sampling)
    feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

    fesnaps, = solution_snapshots(rbsolver,feop)
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

    println(perf)
  end

end

main(:pod)
main(:ttsvd)

end

using Gridap
using GridapROMs
using DrWatson

using GridapROMs.RBSteady

dir = datadir("paper_results")

for test in ("2d_poisson","3d_poisson")
  M = test == "2d_poisson" ? (250,350,460) : (40,50,60)
  for method in ("pod",)#,"ttsvd"
    for m in M
      dirm = joinpath(dir,test*"_$(m)_"*method)
      stats = RBSteady.load_stats(dirm)
      println(stats)
    end
  end
end

for test in ("2d_poisson","3d_poisson")
  M = test == "2d_poisson" ? (250,350,460) : (40,50,60)
  for method in ("ttsvd",)#,"ttsvd"
    for m in M
      for tol in (1e-2,1e-3,1e-4)
        dirm = joinpath(joinpath(dir,test*"_$(m)_"*method),"$tol")
        basis = RBSteady.load_projection(dirm;label="test")
        n = num_reduced_dofs(basis)
        println("ndofs at path = $dirm are: $n")
      end
    end
  end
end

using Gridap
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:ttsvd
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"
pdomain = (1,10,1,10,1,10)

domain = (0,1,0,1)
partition = (50,50)
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

tolrank = tol_or_rank(tol,rank)
if method == :pod
  state_reduction = PODReduction(tolrank,energy;nparams,sketch)
else method == :ttsvd
  tolranks = fill(tolrank,3)
  state_reduction = Reduction(tolranks,energy;nparams)
end

fesolver = LUSolver()
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

pspace = ParamSpace(pdomain)
feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
fesnaps, = solution_snapshots(rbsolver,feop)

using LinearAlgebra
using GridapROMs.RBSteady
using GridapROMs.TProduct

X = assemble_matrix(feop,energy)
X1 = kron(X)
X2 = kron(Rank1Tensor([X[2][1],X[1][2]]))
X3 = kron(Rank1Tensor([X[1][1]+X[2][1],X[1][2]+X[2][2]]))
X4 = kron(X[1])

A = reshape(fesnaps,:,nparams)
A1 = X1*A
A2 = X2*A
A3 = X3*A
A4 = X4*A

using LowRankApprox
U1,S1,V1 = psvd(A1)
U2,S2,V2 = psvd(A2)
U3,S3,V3 = psvd(A3)
U4,S4,V4 = psvd(A4)

e1 = cumsum(S1.^2;dims=1)
r1 = findfirst(e1 .>= (1-tol^2)*e1[end])
e2 = cumsum(S2.^2;dims=1)
r2 = findfirst(e2 .>= (1-tol^2)*e2[end])
e3 = cumsum(S3.^2;dims=1)
r3 = findfirst(e3 .>= (1-tol^2)*e3[end])
e4 = cumsum(S4.^2;dims=1)
r4 = findfirst(e4 .>= (1-tol^2)*e4[end])
