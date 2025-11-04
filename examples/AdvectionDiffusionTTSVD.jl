module AdvectionDiffusionTTSVD

include("ExamplesInterface.jl")

pdomain = (1,10,1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (40,40)
model = TProductDiscreteModel(domain,partition)

Ω = Triangulation(model)
Γn = BoundaryTriangulation(model,tags=[8])

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterize(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterize(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterize(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterize(h,μ)

b(μ) = x -> VectorValue(0.0,μ[4])
bμ(μ) = parameterize(b,μ)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
trial = ParamTrialFESpace(test,gμ)

degree = 2*order
dΩ = Measure(Ω,degree)
dΓn = Measure(Γn,degree)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u) + v*(bμ(μ)⋅∇(u)))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

fesolver = LUSolver()

tol = fill(1e-5,4)
energy(du,v) = ∫(v⋅du)dΩ + ∫(∇(v)⊙∇(du))dΩ
state_reduction = Reduction(tol,energy;nparams=80)
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=40,nparams_jac=40)

dir = datadir("adv_diff_ttsvd")
create_dir(dir)

tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
run_test(dir,rbsolver,feop,tols)

end
