module MovingElasticity3D

using DrWatson
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Arrays
using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.Uncommon
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

import GridapROMs.RBSteady: load_stats

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:ttsvd
n=30
tol=1e-4
rank=nothing
nparams = 100

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.3,0.7)
pspace = ParamSpace(pdomain)

pmin = Point(-1,-1,-1)
pmax = Point(1,1,1)
dp = pmax - pmin

partition = (n,n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

g(x) = VectorValue(x[1],-x[2],0.0)

# dimensional
# λ = 3.5e8*0.33/((1+0.33)*(1-2*0.33))
# p = 3.5e8/(2(1+0.33))
# σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

# adimensional
λ = 0.33/((1+0.33)*(1-2*0.33))
p = 1/(2(1+0.33))
σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

order = 1
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
testbg = FESpace(Ωbg,reffe,conformity=:H1)

energy(du,v) = ∫(v⋅du)dΩbg + ∫(∇(v)⊙∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,4) : tolrank
ncentroids = 4
ncentroids_res = ncentroids_jac = 4
fesolver = ExtensionSolver(LUSolver())

const γd = 10.0
const hd = dp[1]/n

function rb_solver(tolrank,nparams)
  tolrankhr = tolrank.*1e-2
  state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)
  residual_reduction = LocalHyperReduction(tolrankhr;nparams,ncentroids=ncentroids_res,interp=true)
  jacobian_reduction = LocalHyperReduction(tolrankhr;nparams,ncentroids=ncentroids_jac,interp=true)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function global_rb_solver(tolrank,nparams)
  tolrankhr = tolrank.*1e-2
  state_reduction = Reduction(tolrank,energy;nparams)
  residual_reduction = RBFHyperReduction(tolrankhr;nparams)
  jacobian_reduction = RBFHyperReduction(tolrankhr;nparams)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function def_fe_operator(μ)
  geo = popcorn(r0=μ[1])
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v,dΩ,dΓ) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⊙u )dΓ
  l(v,dΓ) = ∫( (γd/hd)*v⋅g - (n_Γ⋅(σ∘ε(v)))⋅g )dΓ
  res(u,v,dΩ,dΓ) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(v,dΓ)

  domains = FEDomains((Ω,Γ),(Ω,Γ))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

function get_trians(μ)
  geo = popcorn(r0=μ[1])
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
  Γ = EmbeddedBoundary(cutgeo)

  return Ω,Ωout,Γ
end

function compute_err(x,x̂,μ)
  geo = popcorn(r0=μ[1])
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

  energy(u,v) = ∫(v⋅u)dΩbg + ∫(∇(v)⊙∇(u))dΩbg + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))⋅u )dΓ
  X = assemble_matrix(energy,testagg,testagg)

  d2bgd = Extensions.get_fdof_to_bg_fdof(testbg,testagg)
  xd = x[d2bgd]
  x̂d = x̂[d2bgd]

  Y = (assemble_matrix((u,v)->∫( v⋅u + ∇(v)⊙∇(u) )dΩbg,testbg,testbg))[d2bgd,d2bgd]

  errH1 = compute_relative_error(xd,x̂d,X)
  _errH1 = compute_relative_error(xd,x̂d,Y)
  errl2 = compute_relative_error(xd,x̂d)
  return (errH1,_errH1,errl2)
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
    u,û,e,bgΩ = vec(x[:,:,:,:,i]),vec(x̂[:,:,:,:,i]),abs.(vec(x[:,:,:,:,i]-x̂[:,:,:,:,i])),Ωbg.trian
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

  error = [0.0,0.0,0.0]
  for (i,μ) in enumerate(μon)
    if method==:ttsvd
      xi,x̂i = vec(fesnaps[:,:,:,:,i]),vec(rbsnaps[:,:,:,:,i])
    else
      xi,x̂i = fesnaps[:,i],rbsnaps[:,i]
    end
    error .+= compute_err(xi,x̂i,μ)
  end
  error ./= param_length(μon)

  speedup = compute_speedup(festats,rbstats)
  map(e -> ROMPerformance(e,speedup),error)
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

test_dir = datadir("moving_elasticity")
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

rbsolver = rb_solver(tolrank,nparams)

try
  fesnaps = load_snapshots(test_dir)
  x = load_snapshots(test_dir;label="online")
  festats = load_stats(test_dir;label="online")
catch
  fesnaps, = solution_snapshots(rbsolver,feop,μ)
  x,festats = solution_snapshots(rbsolver,feopon,μon)
  save(test_dir,fesnaps)
  save(test_dir,x;label="online")
  save(test_dir,festats;label="online")
end

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

perfsH1 = ROMPerformance[]
_perfsH1 = ROMPerformance[]
perfsl2 = ROMPerformance[]
maxsizes = Vector{Int}[]

tols = (1e-0,1e-1,1e-2,1e-3)

for (itol,tol) in enumerate(tols)
  tolrank = tol_or_rank(tol,rank)
  tolrank = method == :ttsvd ? fill(tolrank,4) : tolrank
  rbsolver = rb_solver(tolrank,nparams)

  dir = joinpath(test_dir,string(tol))
  create_dir(dir)

  rbop = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  if itol > 1
    perfH1,_perfH1,perfl2 = postprocess(dir,rbsolver,feopon,rbop,x,x̂,festats,rbstats)
    maxsize = max_subspace_size(rbop)

    println(μon[1])
    println(perfH1)
    println(_perfsH1)
    push!(perfsH1,perfH1)
    push!(_perfsH1,_perfH1)
    push!(perfsl2,perfl2)
    push!(maxsizes,maxsize)
  end
end

serialize(joinpath(test_dir,"resultsH1"),perfsH1)
serialize(joinpath(test_dir,"_resultsH1"),_perfsH1)
serialize(joinpath(test_dir,"resultsl2"),perfsl2)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)

deserialize(joinpath(test_dir,"resultsH1"))
deserialize(joinpath(test_dir,"maxsizes"))

end
