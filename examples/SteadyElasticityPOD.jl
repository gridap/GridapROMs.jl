module SteadyElasticityPOD

using DrWatson
using Gridap
using GridapROMs

include("ExamplesInterface.jl")

pdomain = (1e10,9*1e10,0.25,0.42,-4*1e5,4*1e5,-4*1e5,4*1e5,-4*1e5,4*1e5)
pspace = ParamSpace(pdomain)

domain = (0,1,0,0.5,0,0.25)
partition = (20,10,5)
model = CartesianDiscreteModel(domain,partition)
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"dirichlet",[1,3,5,7,13,15,17,19,25])
add_tag_from_tags!(labels,"neumann1",[22])
add_tag_from_tags!(labels,"neumann2",[24])
add_tag_from_tags!(labels,"neumann3",[26])

Ω = Triangulation(model)
Γ1 = BoundaryTriangulation(model,tags="neumann1")
Γ2 = BoundaryTriangulation(model,tags="neumann2")
Γ3 = BoundaryTriangulation(model,tags="neumann3")

λ(μ) = μ[1]*μ[2]/((1+μ[2])*(1-2*μ[2]))
p(μ) = μ[1]/(2(1+μ[2]))

σ(μ) = ε -> λ(μ)*tr(ε)*one(ε) + 2*p(μ)*ε
σμ(μ) = ParamFunction(σ,μ)

h1(μ) = x -> VectorValue(0.0,0.0,μ[3])
h1μ(μ) = ParamFunction(h1,μ)

h2(μ) = x -> VectorValue(0.0,μ[4],0.0)
h2μ(μ) = ParamFunction(h2,μ)

h3(μ) = x -> VectorValue(μ[5]*x[1],0.0,0.0)
h3μ(μ) = ParamFunction(h3,μ)

g(μ) = x -> VectorValue(0.0,0.0,0.0)
gμ(μ) = ParamFunction(g,μ)

order = 2
reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags="dirichlet")
trial = ParamTrialFESpace(test,gμ)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓ1 = Measure(Γ1,degree)
dΓ2 = Measure(Γ2,degree)
dΓ3 = Measure(Γ3,degree)

a(μ,u,v,dΩ) = ∫( ε(v) ⊙ (σμ(μ)∘ε(u)) )*dΩ
l(μ,u,v,dΓ1,dΓ2,dΓ3) = ∫(v⋅h1μ(μ))dΓ1 + ∫(v⋅h2μ(μ))dΓ2 + ∫(v⋅h3μ(μ))dΓ3

trian_l,trian_a = (Γ1,Γ2,Γ3),(Ω,)
domains = FEDomains(trian_l,trian_a)
feop = LinearParamOperator(l,a,pspace,trial,test,domains)

fesolver = LUSolver()

energy(du,v) = ∫(∇(v)⊙∇(du))dΩ

tol = 1e-5
state_reduction = Reduction(tol,energy;nparams=80,sketch=:sprn)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=40,nparams_jac=40)

dir = datadir("elasticity_pod")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols)

end
