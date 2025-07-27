module ParamMappedModelTest

using GridapROMs
using Gridap
using Gridap.Geometry
using Test

domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

μ = Realization([[1.0],[2.0]])
ϕ(μ) = x->VectorValue(x[2],μ[1]*x[1])
ϕμ(μ) = parameterize(ϕ,μ)
mmodel = MappedDiscreteModel(model,ϕμ(μ))

Ωm = Triangulation(mmodel)
Γm = BoundaryTriangulation(mmodel,tags=8)

dΩm = Measure(Ωm,4)
dΓm = Measure(Γm,4)

g(μ) = x->x[1]+μ[1]*x[2]
gμ(μ) = parameterize(g,μ)

reffe = ReferenceFE(lagrangian,Float64,2)
Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Um = ParamTrialFESpace(Vm,gμ)

Umμ = Um(μ)

ν(μ) = x->x[1]+μ[1]*x[2]
νμ(μ) = parameterize(ν,μ)
f(μ) = x->x[1]+μ[1]*x[2]
fμ(μ) = parameterize(f,μ)

am(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm
bm(μ,u,v) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm - ∫(fμ(μ)*v)dΩm #+ ∫(fμ(μ)*v)dΓm

pspace = ParamSpace((1.0,2.0))
opm = LinearParamOperator(bm,am,pspace,Um,Vm)

xm, = solve(LUSolver(),opm,μ)

function gridap_solution(μ)
  mmodel = MappedDiscreteModel(model,ϕ(μ))
  Ωm = Triangulation(mmodel)
  dΩm = Measure(Ωm,4)
  Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  Um = TrialFESpace(Vm,g(μ))
  am(u,v) = ∫(ν(μ)*∇(v)⋅∇(u))dΩm
  bm(v) = ∫(f(μ)*v)dΩm #+ ∫(f(μ)*v)dΓm
  opm = AffineFEOperator(am,bm,Um,Vm)
  xm = solve(LUSolver(),opm)
  get_free_dof_values(xm)
end

for (i,μi) in enumerate([1.0,2.0])
  @test xm[i] ≈ gridap_solution(μi)
end

end
