# module MovingPoisson

using DrWatson
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.Uncommon
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

method=:ttsvd
compression=:local
sketch=:sprn
n=40
tol=1e-4
rank=nothing
nparams = 300

pdomain = (0.5,1.5,0.25,0.35)
pspace = ParamSpace(pdomain)

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

order = 1
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1)

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
ncentroids = 1
ncentroids_res = ncentroids_jac = 1
fesolver = ExtensionSolver(LUSolver())

const γd = 10.0
const hd = dp[1]/n

function rb_solver(tol,nparams)
  tol = method == :ttsvd ? fill(tol,3) : tol
  tolhr = tol.*1e-2
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  residual_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=ncentroids_res)
  jacobian_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=ncentroids_jac)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(μ[2],x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  # a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  # l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  # res(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ,dΓ)

  # domains = FEDomains((Ω,Γ),(Ω,Γ))
  a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v) = ∫(∇(v)⋅∇(u))dΩ - l(v)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test)
end

function get_trians(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(μ[2],x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
  Γ = EmbeddedBoundary(cutgeo)

  return Ω,Ωout,Γ
end

function compute_err(x,x̂,μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(μ[2],x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)
  n_Γ = get_normal_vector(Γ)

  order = 2
  degree = 2*order

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  energy(u,v) = ∫( v*u + ∇(v)⋅∇(u) )dΩ + ∫( (γd/hd)*v*u )dΓ
  X = assemble_matrix(energy,testagg,testagg)

  d2bgd = Extensions.get_fdof_to_bg_fdof(testbg,testagg)
  xd = x[d2bgd]
  x̂d = x̂[d2bgd]

  Y = (assemble_matrix((u,v)->∫( v*u + ∇(v)⋅∇(u) )dΩbg,testbg,testbg))[d2bgd,d2bgd]

  errH1 = compute_relative_error(xd,x̂d,X)
  return errH1
end

function RBSteady.reduced_operator(rbsolver::RBSolver,feop::ParamOperator,sol,jac,res)
  red_trial,red_test = reduced_spaces(rbsolver,feop,sol)
  jac_red = RBSteady.get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = RBSteady.get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function plot_sol(rbop,x,x̂,μon,dir,i=1)
  if method==:ttsvd
    u,û,e,bgΩ = vec(x[:,:,i]),vec(x̂[:,:,i]),abs.(vec(x[:,:,i]-x̂[:,:,i])),Ωbg.trian
  else
    u,û,e,bgΩ = x[:,i],x̂[:,i],abs.(x[:,i]-x̂[:,i]),Ωbg
  end
  uh = FEFunction(testbg,u)
  ûh = FEFunction(testbg,û)
  eh = FEFunction(testbg,e)

  Ω,Ωout, = get_trians(μon.params[i])

  writevtk(bgΩ,joinpath(dir,"bgtrian"),cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
  writevtk(Ω,joinpath(dir,"trian"),cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
  writevtk(Ωout,joinpath(dir,"trianout"),cellfields=["uh"=>uh,"ûh"=>ûh,"eh"=>eh])
end

function postprocess(
  dir,
  solver,
  feop,
  rbop,
  fesnaps,
  x̂,
  festats,
  rbstats)

  μon = get_realization(fesnaps)
  rbsnaps = RBSteady.to_snapshots(rbop,x̂,μon)
  plot_sol(rbop,fesnaps,rbsnaps,μon,dir)

  error = 0.0
  for (i,μ) in enumerate(μon)
    if method==:ttsvd
      xi,x̂i = vec(fesnaps[:,:,i]),vec(rbsnaps[:,:,i])
    else
      xi,x̂i = fesnaps[:,i],rbsnaps[:,i]
    end
    error += compute_err(xi,x̂i,μ)
  end
  error /= param_length(μon)

  speedup = compute_speedup(festats,rbstats)
  ROMPerformance(error,speedup)
end

function max_subspace_size(rbop::LocalRBOperator)
  N = max_subspace_size(rbop.test)
  Qa = max_subspace_size(rbop.lhs)
  Qf = max_subspace_size(rbop.rhs)
  return [N,Qa,Qf]
end

max_subspace_size(r::RBSpace) = max_subspace_size(r.subspace)
max_subspace_size(a::Projection) = num_reduced_dofs(a)
max_subspace_size(a::NormedProjection) = max_subspace_size(a.projection)
max_subspace_size(a::AffineContribution) = maximum(map(max_subspace_size,a.values))

function max_subspace_size(a::LocalProjection)
  maxsize = 0
  for proj in a.projections
    maxsize = max(maxsize,max_subspace_size(proj))
  end
  return maxsize
end

test_dir = datadir("moving_poisson")
create_dir(test_dir)

μ = realization(pspace;nparams)
feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = param_operator(μon) do μ
  println("------------------")
  def_fe_operator(μ)
end

for opon in feopon.operators
  println(num_free_dofs(get_test(opon)))
end

rbsolver = rb_solver(tol,nparams)
fesnaps, = solution_snapshots(rbsolver,feop)
x,festats = solution_snapshots(rbsolver,feopon,μon)

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

perfsH1 = ROMPerformance[]
maxsizes = Vector{Int}[]

for tol in (1e-0,1e-1,1e-2,1e-3,1e-4)
  rbsolver = rb_solver(tol,nparams)

  dir = joinpath(test_dir,string(tol))
  create_dir(dir)

  rbop′ = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
  rbop = change_operator(rbop′,feopon)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  perfH1 = postprocess(dir,rbsolver,feopon,rbop,x,x̂,festats,rbstats)
  maxsize = max_subspace_size(rbop)

  println(μon[1])
  push!(perfsH1,perfH1)
  push!(maxsizes,maxsize)
end

serialize(joinpath(test_dir,"resultsH1"),perfsH1)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)

# end

rbop′ = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
rbop1′ = get_local(rbop′,first(μ))
feopon1 = feopon[first(μon)]
rbop = change_operator(rbop1′,feopon1)

opμ = rbop
r = RBSteady._to_realization(μon,first(μon))
# x̂, = solve(rbsolver,opμ,r)
U = get_trial(opμ)(r)
x̂ = zero_free_values(U)
nlop = parameterize(opμ,r)
syscache = allocate_systemcache(nlop,x̂)

b = syscache.b
fill!(b,zero(eltype(b)))

uh = EvaluationFunction(nlop.paramcache.trial,x̂)
V = get_test(opμ)
v = get_fe_basis(V)

rhs = RBSteady.get_rhs(opμ)
bg_trian = first(get_domains(rhs))
dc = get_res(opμ.op)(r,uh,v)
strian = [get_domains(dc)...][2]

rhs_strian = move_interpolation(rhs[bg_trian],V,strian)
vecdata = RBSteady.collect_reduced_cell_hr_vector(V,dc,strian,rhs_strian)
assemble_hr_array_add!(b.fecache[bg_trian],vecdata...)

cell_vec_r,cell_idofs,icells = vecdata

rows = rhs[bg_trian].interpolation.domain.metadata
cells = RBSteady.reduced_cells(V,strian,rows)
irowcols = RBSteady.reduced_irows(V,strian,cells,rows)