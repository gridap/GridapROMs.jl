using Gridap
using GridapROMs
using GridapROMs.TProduct

domain = (0,1,0,1)
partition = (2,2)
model = TProductDiscreteModel(domain,partition)

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)

u(x) = x[1] - x[2]
f(x) = -Δ(u)(x)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ
l(v) = ∫(f*v)dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags="boundary")
trial = TrialFESpace(test,u)

op = AffineFEOperator(a,l,trial,test)
uh = solve(op)

# this is only the free dof ids
# dof_map = get_dof_map(test)
# this is both the free and dirichlet dof ids
fddof_map = TProduct.get_dof_map_with_diri(test)

vals = zeros(size(fddof_map))
for (k,idofk) in enumerate(fddof_map)
  vals[k] = idofk>0 ? uh.free_values[idofk] : uh.dirichlet_values[-idofk]
end
