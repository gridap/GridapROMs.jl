module MovingPoissonNeumann


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

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:ttsvd
n=20
tol=1e-4
rank=nothing
Ntop,Nbot = 400,200

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.5,1.5,0.25,0.35)
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
g(x) = 0.0

order = 2
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1,dirichlet_tags=[1,3,7])

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
μ = realization(pspace;nparams=Ntop)
ncentroids = 16
ncentroids_res = ncentroids_jac = 8
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

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(μ[2],x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ
  l(v,dΩ) = ∫(f⋅v)dΩ
  res(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ)

  domains = FEDomains((Ω,),(Ω,))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
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
  testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
  testagg = AgFEMSpace(testact,aggregates)

  energy(u,v) = ∫( v*u + ∇(v)⋅∇(u) )dΩ
  X = assemble_matrix(energy,testagg,testagg)

  d2bgd = Extensions.get_fdof_to_bg_fdof(testbg,testagg)
  xd = x[d2bgd]
  x̂d = x̂[d2bgd]

  compute_relative_error(xd,x̂d,X)
end

function RBSteady.reduced_operator(rbsolver::RBSolver,feop::ParamOperator,sol,jac,res)
  red_trial,red_test = reduced_spaces(rbsolver,feop,sol)
  jac_red = RBSteady.get_jacobian_reduction(rbsolver)
  red_lhs = reduced_jacobian(jac_red,red_trial,red_test,jac)
  res_red = RBSteady.get_residual_reduction(rbsolver)
  red_rhs = reduced_residual(res_red,red_test,res)
  RBOperator(feop,red_trial,red_test,red_lhs,red_rhs)
end

function plot_sol(rbop,x,x̂,μon,dir,i=num_params(μon))

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

test_dir = datadir("moving_poisson_neumann")
create_dir(test_dir)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

μon = realization(pspace;nparams=10,sampling=:uniform)
feopon = param_operator(μon) do μ
  println("------------------")
  def_fe_operator(μ)
end

rbsolver = rb_solver(tolrank,Ntop)
fesnaps, = solution_snapshots(rbsolver,feop)
x,festats = solution_snapshots(rbsolver,feopon,μon)

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

perfs = ROMPerformance[]
maxsizes = Vector{Int}[]

for nparams in Ntop:-10:Nbot
  if nparams == Ntop
    _feop,_rbsolver,_fesnaps,_jacs,_ress = feop,rbsolver,fesnaps,jacs,ress
  else
    _feop = Uncommon._get_at_param(feop,μ[1:nparams])
    _rbsolver = rb_solver(tolrank,nparams)
    _fesnaps = select_snapshots(fesnaps,1:nparams)
    _jacs,_ress = select_snapshots(jacs,1:nparams),select_snapshots(ress,1:nparams)
  end

  dir = joinpath(test_dir,string(nparams))
  create_dir(dir)

  rbop = reduced_operator(_rbsolver,_feop,_fesnaps,_jacs,_ress)
  x̂,rbstats = solve(_rbsolver,rbop,μon)

  perf = postprocess(dir,_rbsolver,feopon,rbop,x,x̂,festats,rbstats)
  maxsize = max_subspace_size(rbop)
  println(perf)
  push!(perfs,perf)
  push!(maxsizes,maxsize)
end

serialize(joinpath(test_dir,"results"),perfs)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)

end

# using Plots
# maxs = deserialize(joinpath(test_dir,"maxsizes"))
# perfs = deserialize(joinpath(test_dir,"results"))
# N = first.(maxs)
# errs = map(x -> x.error,perfs)
# plot(N,reshape(errs,:,1))
