module HeatEquationSTRB

using DrWatson
using Gridap
using GridapROMs

include("ExamplesInterface.jl")

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 40*dt

pdomain = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

model_dir = datadir(joinpath("models","model_circle_h007.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"neumann",["outlet"])
add_tag_from_tags!(labels,"dirichlet",["inlet","walls"])

Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags="neumann")

ν(x,μ,t) = 1+exp(sin(t)*x[1]/sum(μ))
ν(μ,t) = x->ν(x,μ,t)
νμt(μ,t) = TransientParamFunction(ν,μ,t)

f(x,μ,t) = 1.
f(μ,t) = x->f(x,μ,t)
fμt(μ,t) = TransientParamFunction(f,μ,t)

h(x,μ,t) = abs(cos(t/μ[3]))
h(μ,t) = x->h(x,μ,t)
hμt(μ,t) = TransientParamFunction(h,μ,t)

g(x,μ,t) = μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
g(μ,t) = x->g(x,μ,t)
gμt(μ,t) = TransientParamFunction(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = ParamFunction(u0,μ)

order = 1
degree = 2*order+1
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

stiffness(μ,t,u,v,dΩ) = ∫(νμt(μ,t)*∇(v)⋅∇(u))dΩ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

energy(du,v) = ∫(du*v)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags=["dirichlet"])
trial = TransientTrialParamFESpace(test,gμt)
feop = TransientLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(LUSolver(),dt,θ)

tol = 1e-5
energy(du,v) = ∫(∇(v)⊙∇(du))dΩ
state_reduction = HighOrderReduction(tol,energy;nparams=40,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=1)

dir = datadir("heateq_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
ExamplesInterface.run_test(dir,rbsolver,feop,tols,uh0μ)

end
