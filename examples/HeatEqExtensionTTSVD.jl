module HeatEqExtensionTTSVD

include("ExamplesInterface.jl")

θ = 0.5
dt = 0.0025
t0 = 0.0
tf = 40*dt

pdomain = fill([1,10],3)
tdomain = t0:dt:tf
ptspace = TransientParamSpace(pdomain,tdomain)

const L = 2
const W = 2
const n = 40
const γd = 10.0
const hd = max(L,W)/n

domain = (0,L,0,W)
partition = (n,n)
bgmodel = TProductDiscreteModel(domain,partition)

geo = !disk(0.3,x0=Point(1.0,1.0))
cutgeo = cut(bgmodel,geo)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)
Γn = BoundaryTriangulation(model,tags=8)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)
dΓn = Measure(Γn,degree)

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

ν(μ,t) = x -> (1+t)*μ[1]
νμt(μ,t) = parameterize(ν,μ,t)

f(μ,t) = x -> 1.
fμt(μ,t) = parameterize(f,μ,t)

h(μ,t) = x -> abs(cos(t/μ[2]))
hμt(μ,t) = parameterize(h,μ,t)

g(μ,t) = x -> abs(sin(t/μ[3]))
gμt(μ,t) = parameterize(g,μ,t)

u0(μ) = x -> 0.0
u0μ(μ) = parameterize(u0,μ)

order = 1
degree = 2*order
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

stiffness(μ,t,u,v,dΩ) = ∫(νμt(μ,t)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
trian_mass = (Ω,)
domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testact = FESpace(Ωact,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testagg = AgFEMSpace(testact,aggregates)
test = DirectSumFESpace(testbg,test)
trial = TransientTrialParamFESpace(testext,gμt)

feop = TransientExtensionLinearParamOperator((stiffness,mass),res,ptspace,trial,test,domains)
uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

fesolver = ThetaMethod(ExtensionSolver(LUSolver()),dt,θ)

tol = 1e-5
energy(du,v) = ∫(∇(v)⋅∇(du))dΩbg
state_reduction = HighDimReduction(tol,energy;nparams=40,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=20,nparams_jac=20,nparams_djac=1)

dir = datadir("heateq_ttsvd")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols,uh0μ)

end
