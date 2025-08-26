using Gridap
using GridapROMs

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
fddof_map = get_dof_map_with_diri(test)

vals = zeros(size(fddof_map))
for (k,idofk) in enumerate(fddof_map)
  vals[k] = idofk>0 ? uh.free_values[idofk] : uh.dirichlet_values[-idofk]
end

# with components

ucomp(x) = VectorValue(x[1],-x[2])
fcomp(x) = -Δ(ucomp)(x)

acomp(u,v) = ∫(∇(v)⊙∇(u))dΩ
lcomp(v) = ∫(fcomp⋅v)dΩ

reffecomp = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
testcomp = TestFESpace(Ω,reffecomp;conformity=:H1,dirichlet_tags="boundary")
trialcomp = TrialFESpace(testcomp,ucomp)

opcomp = AffineFEOperator(acomp,lcomp,trialcomp,testcomp)
uhcomp = solve(opcomp)

fddofcomp_map = get_dof_map_with_diri(testcomp)

valscomp = zeros(size(fddofcomp_map))
for (k,idofk) in enumerate(fddofcomp_map)
  valscomp[k] = idofk>0 ? uhcomp.free_values[idofk] : uhcomp.dirichlet_values[-idofk]
end

# with unfitted

using GridapEmbedded
using GridapROMs.Extensions

R  = 0.5
L  = 0.5*R
p1 = Point(0.0,0.0)
p2 = p1 + VectorValue(-L,L)

geo1 = disk(R,x0=p1)
geo2 = disk(R,x0=p2)
geo3 = setdiff(geo1,geo2)

t = 1.01
pmin = p1-t*R
pmax = p1+t*R

n = 30
partition = (n,n)
bgmodel = TProductDiscreteModel(pmin,pmax,partition)
dp = pmax - pmin

cutgeo = cut(bgmodel,geo3)

Ω_act = Triangulation(cutgeo,ACTIVE)
Ω_bg = Triangulation(bgmodel)
Ω = Triangulation(cutgeo,PHYSICAL)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
Vstd = TestFESpace(Ω_act,reffe,conformity=:H1)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

testagg = AgFEMSpace(Vstd,aggregates)

Vbg = FESpace(Ω_bg,reffe,conformity=:H1)
V = DirectSumFESpace(Vbg,testagg)
U = TrialFESpace(V,u)

degree = 2*order
dΩ = Measure(Ω,degree)

Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)
dΓ = Measure(Γ,degree)

u(x) = x[1] - x[2] # Solution of the problem
const γd = 10.0    # Nitsche coefficient
const h = dp[1]/n  # Mesh size according to the parameters of the background grid

a(u,v) =
  ∫( ∇(v)⋅∇(u) )dΩ +
  ∫( (γd/h)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ

l(v) = ∫( (γd/h)*v*u - (n_Γ⋅∇(v))*u )dΓ
# careful: in my code, we provide the residual instead of the RHS
res(u,v) = ∫( ∇(v)⋅∇(u) )dΩ - l(v)

op = ExtensionLinearOperator(res,a,U,V)

# method 1: pad by zero
# careful: my code returns a vector of values, not a FE function. In this example
# it's ok because there are no Dirichlet conditions
solver = ExtensionSolver(LUSolver(),ZeroExtension())
uvals = solve(solver,op)

# # method 2: harmonic extension
# solver = ExtensionSolver(LUSolver(),HarmonicExtension())
# uh = solve(solver,op)

fddof_map = get_dof_map_with_diri(V)

vals = zeros(size(fddof_map))
for (k,idofk) in enumerate(fddof_map)
  @assert idofk>0
  vals[k] = uvals[idofk]
end
