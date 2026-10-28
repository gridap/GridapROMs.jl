module NStokesTransientSTRB

using DrWatson
using Gridap
using Gridap.MultiField
using Serialization
using Test

using GridapROMs
using GridapROMs.DofMaps
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.RBTransient

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers

using Plots

θ = 1.0
dt = 0.0025
t0 = 0.0
tf = 30*dt

pdomain = (1,10,1,10,1,2)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

model_dir = datadir(joinpath("models","back_facing_channel.json"))
model = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(model)

order = 2
degree = 2*order+1
Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

const Re = 100.0
a(x,μ,t) = μ[1]/Re
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = parameterise(a,μ,t)

f(x) = VectorValue(0.0,0.0)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v) = ∫( v⊙(conv∘(u,∇(u))) )dΩ
dc(u,du,v) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩ

const Lb = 0.2
const Ub = 0.4
inflow(μ,t) = abs(1-cos(2π*t/tf)+sin((2π*t/tf)/μ[2])/μ[2])
g_in(x,μ,t) = VectorValue(μ[3]*(x[2]-Ub)*(x[2]-Lb)*inflow(μ,t),0.0)
g_in(μ,t) = x->g_in(x,μ,t)
gμt_in(μ,t) = parameterise(g_in,μ,t)
g_0(x,μ,t) = VectorValue(0.0,0.0)
g_0(μ,t) = x->g_0(x,μ,t)
gμt_0(μ,t) = parameterise(g_0,μ,t)

u0(μ) = x -> VectorValue(0.0,0.0)
u0μ(μ) = parameterise(u0,μ)
p0(μ) = x -> 0.0
p0μ(μ) = parameterise(p0,μ)

rhs(μ,t,(v,q)) = ∫(v⋅f)dΩ
stiffness(μ,t,(u,p),(v,q)) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
mass(μ,t,(uₜ,pₜ),(v,q)) = ∫(v⋅uₜ)dΩ

res_nlin(μ,t,(u,p),(v,q)) = c(u,v)
jac_nlin(μ,t,(u,p),(du,dp),(v,q)) = dc(u,du,v)

trian_res = (Ω,)
trian_jac = (Ω,)
trian_jac_t = (Ω,)
domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
domains_nlin = FEDomains(trian_res,(trian_jac,))

coupling = DivCoupling()
energy_u(du,v) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ
energy_p(dp,q) = ∫(dp*q)dΩ
energy = BlockNorm((EnergyNorm(energy_u),EnergyNorm(energy_p)))

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(model,reffe_u;conformity=:H1,dirichlet_tags=["inlet","walls"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(model,reffe_p;conformity=:H1)
trial_p = TransientTrialParamFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

res_bundle = Bundle(rhs,res_nlin)
jacs_bundle = (Bundle(stiffness,jac_nlin),Bundle(mass,nothing))
domains_bundle = Bundle(domains_lin,domains_nlin)
feop = TransientParamOperator(res_bundle,jacs_bundle,ptspace,trial,test,domains_bundle)

fesolver = ThetaMethod(NewtonSolver(LUSolver();rtol=1e-10,maxiter=20,verbose=true),dt,θ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=60)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jacs=(20,1))

dir = datadir("transient_nstokes_pod")
mkpath(dir)

tols = [1e-4,]
run_test(dir,rbsolver,feop,tols,xh0μ)

end
