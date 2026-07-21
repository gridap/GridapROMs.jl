using Gridap
using GridapROMs

using GridapROMs.RBSteady
using GridapROMs.Nonlinear

method=:pod
compression=:global
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (10,10)
model = CartesianDiscreteModel(domain,partition)

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

fesolver = LUSolver()
state_reduction = Reduction(tol,energy;nparams,sketch,compression)
res_reduction = NNHyperReduction(tol;nparams_res,sketch,compression)
jac_reduction = NNHyperReduction(tol;nparams_jac,sketch,compression)
rbsolver = RBSolver(fesolver,state_reduction,res_reduction,jac_reduction)

feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
fesnaps, = solution_snapshots(rbsolver,feop)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realisation(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

# 

res_reduction_2 = NNOperatorReduction(tol;nparams=nparams_res)
jac_reduction_2 = NNOperatorReduction(tol;nparams=nparams_jac)
rbsolver_2 = RBSolver(fesolver,state_reduction,res_reduction_2,jac_reduction_2)

rbop_2 = reduced_operator(rbsolver_2,feop,fesnaps)

x̂_2,rbstats_2 = solve(rbsolver_2,rbop_2,μon)
perf_2 = eval_performance(rbsolver_2,rbop_2,x,x̂_2,festats,rbstats_2)