using DrWatson
using LinearAlgebra
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.ParamAlgebra
using GridapROMs.Utils

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=20
tol=1e-4
rank=nothing
nparams=250
nparams_res=250
nparams_jac=250

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.5,1.5)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(2,2)
dp = pmax - pmin

partition = (n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

f(x) = 1.0
g(x) = x[1]-x[2]

order = 2
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags=[1,3,7])

trian_a = (Ωbg,)
trian_res = (Ωbg,)
domains = FEDomains(trian_res,trian_a)

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
ncentroids = 16
state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)

X = assemble_matrix(energy,testbg,testbg)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=nparams,nparams_jac=nparams)

const γd = 10.0
const hd = dp[1]/n

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(R,x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v) = ∫(∇(v)⋅∇(u))dΩ - l(v)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test)
end

μ = realization(pspace;nparams)

feop = param_operator(μ,domains) do μ
  println("------------------")
  def_fe_operator(μ)
end

fesnaps, = solution_snapshots(rbsolver,feop)
red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
red_lhs,red_rhs = RBSteady.reduced_weak_form(rbsolver,feop,red_trial,red_test,fesnaps)
rbop = RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)

μon = realization(pspace;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

xvec = map(x̂,μon) do x̂,μ
  opμ = RBSteady.get_local(rbop,μ)
  trial = RBSteady.get_trial(opμ)
  x = inv_project(trial,x̂)
end

S = stack(xvec)
i = VectorDofMap(size(S,1))
s = Snapshots(S,i,μon)

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

hrres = reduced_residual(rbsolver.residual_reduction,red_test,ress)
hrjac = reduced_jacobian(rbsolver.jacobian_reduction,red_trial,red_test,jacs)

μ = μon[1]
hrresμ = RBSteady.get_local(hrres,μ)[1]

b = allocate_hypred_cache(hrresμ)

fill!(b,zero(eltype(b)))

opμ = def_fe_operator(μ.params)

# U = get_trial(opμ)
# V = get_test(opμ)
U = RBSteady.get_local(red_trial,μ)[1]
V = RBSteady.get_local(red_test,μ)[1]
u = zero_free_values(U)
uh = EvaluationFunction(U,u)
v = get_fe_basis(V)

trian_res = get_domains(hrresμ)

dc = get_res(opμ.op)(uh,v)

strian = trian_res[1]
b_strian = b.fecache[strian]
rhs_strian = hrresμ[strian]
vecdata = collect_cell_hr_vector(V,dc,strian,rhs_strian)
assemble_hr_vector_add!(b_strian,vecdata...)

inv_project!(b,op.rhs)

s = ress[1]
red = rbsolver.residual_reduction.reduction.reduction
basis = projection(red,s)
proj_basis = project(RBSteady.local_values(red_test)[1],basis)
rows,interp = empirical_interpolation(basis)
