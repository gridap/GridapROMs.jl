using DrWatson
using Gridap
using GridapROMs

import GridapROMs.RBTransient: time_combinations

include("../examples/ExamplesInterface.jl")

M = 10
method = :ttsvd

order = 1
degree = 2*order

domain = (0,1,0,1,0,1)
partition = (M,M,M)
model = method==:pod ? CartesianDiscreteModel(domain,partition) : TProductDiscreteModel(domain,partition)
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[26])
dΓn = Measure(Γn,degree)

θ = 0.5
dt = 0.01
t0 = 0.0
tf = 10*dt
tdomain = t0:dt:tf

pdomain = (1,5,1,5,1,5,1,5,1,5,1,5)
ptspace = TransientParamSpace(pdomain,tdomain)

ν(μ,t) = x -> (μ[1]+μ[2]*x[1])#*exp(sin(2pi*t/tf))
νμt(μ,t) = parameterize(ν,μ,t)

f(μ,t) = x -> μ[3]
fμt(μ,t) = parameterize(f,μ,t)

g(μ,t) = x -> μ[4]#exp(-μ[4]*x[2])*(1-cos(2pi*t/tf)+sin(2pi*t/tf)/μ[5])
gμt(μ,t) = parameterize(g,μ,t)

h(μ,t) = x -> μ[6]
hμt(μ,t) = parameterize(h,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = ParamFunction(u0,μ)

stiffness(μ,t,u,v,dΩ) = ∫(νμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

fesolver = ThetaMethod(LUSolver(),dt,θ)

nparams = 50
tol = method==:pod ? 1e-3 : fill(1e-3/sqrt(4),4)
cres,cjac,cdjac = time_combinations(fesolver)

sketch = :sprn
state_reduction = TransientReduction(tol,energy;nparams,sketch)
# res_reduction = TransientMDEIMReduction(cres,TransientReduction(tol;nparams=50))
# jac_reduction = TransientMDEIMReduction(cjac,TransientReduction(tol;nparams=20))
# djac_reduction = TransientMDEIMReduction(cdjac,TransientReduction(tol;nparams=1))

# rbsolver = RBSolver(fesolver,state_reduction,res_reduction,(jac_reduction,djac_reduction))
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=20,nparams_djac=1)

dir = datadir("3d_heateq_$(M)_$(method)")
fesnaps, = ExamplesInterface.try_loading_fe_snapshots(dir,rbsolver,feop,uh0μ)
x,festats,μon = ExamplesInterface.try_loading_online_fe_snapshots(
  dir,rbsolver,feop,uh0μ;reuse_online=true)

using GridapROMs.RBSteady
using GridapROMs.TProduct
red_trial,red_test = RBSteady.old_reduced_spaces(rbsolver,feop,fesnaps)
red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
rbop = reduced_operator(rbsolver,feop,red_trial,red_test,fesnaps)

x̂,rbstats = solve(rbsolver,rbop,μon,uh0μ)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

rbsnaps = RBSteady.to_snapshots(rbop.trial,x̂,μon)
compute_relative_error(x,rbsnaps,get_crossnorm(X))

X = assemble_matrix(feop,energy)
Xk = kron(X)
Xkhat = kron(get_crossnorm(X))
X1 = kron(X[1])

using LowRankApprox
U,S,V = psvd(Matrix(Xk))
U1,S1,V1 = psvd(Matrix(X1))
Uhat,Shat,Vhat = psvd(Matrix(Xkhat))

v = ones(length(S))
v'Xk*v
v'X1*v
v'Xkhat*v

A = permutedims(fesnaps,(1,2,3,5,4))
Ã = reshape(Xk*reshape(A,:,size(A,4)*size(A,5)),size(A))
cores, = ttsvd(state_reduction.reduction.red_style,Ã)
