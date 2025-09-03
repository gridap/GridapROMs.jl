module SteadyStokesPOD

using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs

import Gridap.MultiField: BlockMultiFieldStyle

include("ExamplesInterface.jl")

pdomain = (1,10,-1,5,1,2)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
partition = (40,40)

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
labels = get_face_labeling(bgmodel)
add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])

cutgeo = cut(bgmodel,geo2)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Γ = EmbeddedBoundary(cutgeo)

n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

ν(μ) = x -> μ[1]
νμ(μ) = parameterize(ν,μ)

const W = 1.0
g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(W-x[2]),0.0)*(x[1]==0.0)
gμ(μ) = parameterize(g,μ)

f(μ) = x -> VectorValue(0.0,0.0)
fμ(μ) = parameterize(f,μ)

a(μ,(u,p),(v,q),dΩ,dΓ) = (
  ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ +
  ∫( - v⋅(n_Γ⋅∇(u))*νμ(μ) + (n_Γ⋅∇(v))⋅u*νμ(μ) + (p*n_Γ)⋅v + (q*n_Γ)⋅u )dΓ
)

l(μ,(u,p),(v,q),dΩ) = ∫( νμ(μ)*∇(v)⊙∇(u) - p*(∇⋅(v)) - q*(∇⋅(u)) )dΩ

trian_res = (Ω,)
trian_jac = (Ω,Γ)
domains = FEDomains(trian_res,trian_jac)

coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩbg
energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(dp*q)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫((γd/hd)*du⋅v)dΓ

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ωbg,reffe_u;conformity=:H1,dirichlet_tags="dirichlet")
trial_u = ParamTrialFESpace(test_u,gμ)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ωact,reffe_p;conformity=:H1)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
feop = LinearParamOperator(l,a,pspace,trial,test,domains)

fesolver = LUSolver()

tol = 1e-4
state_reduction = SupremizerReduction(coupling,tol,energy;nparams=100)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=50,nparams_jac=50)

dir = datadir("stokes_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols;trian=Ω)

end
