module MovingStokes

using BlockArrays
using DrWatson
using Serialization

using Gridap
using Gridap.MultiField
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
nparams = 100

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.5,1.5,0.25,0.35)#(0.5,2.5,0.25,0.35)
pspace = ParamSpace(pdomain)

pmin = Point(0,0)
pmax = Point(2,2)#Point(4,2)
dp = pmax - pmin

partition = (n,n)#(2*n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

f(x) = VectorValue(0.0,0.0)
g(x) = VectorValue(x[2]*(2-x[2]),0.0)*(x[1]≈0.0)

order = 2
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
testbg_u = FESpace(Ωbg,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testbg_p = FESpace(Ωbg,reffe_p,conformity=:H1)
trialbg_u = TrialFESpace(testbg_u,g)
trialbg_p = testbg_p

energy((du,dp),(v,q)) = ∫(du⋅v)dΩbg + ∫(dp*q)dΩbg + ∫(∇(v)⊙∇(du))dΩbg
coupling((du,dp),(v,q)) = method==:pod ? ∫(dp*(∇⋅(v)))dΩbg : ∫(dp*∂₁(v))dΩbg + ∫(dp*∂₂(v))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,3) : tolrank
ncentroids = 16
ncentroids_res = ncentroids_jac = 8
extension = BlockExtension([HarmonicExtension(),ZeroExtension()])
fesolver = ExtensionSolver(LUSolver(),extension)

const γd = 10.0
const hd = dp[1]/n

function rb_solver(tol,nparams)
  tolhr = tol.*1e-2
  state_reduction = SupremizerReduction(coupling,tol,energy;nparams,ncentroids,compression=:local)
  residual_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=ncentroids_res,hypred_strategy=:rbf)
  jacobian_reduction = LocalHyperReduction(tolhr;nparams,ncentroids=ncentroids_jac,hypred_strategy=:rbf)
  RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)
end

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1]) #Point(1.0,1.0)#
  geo = !disk(0.35,x0=x0) #!disk(μ[2],x0=x0) #
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  nΓ = get_normal_vector(Γ)

  a((u,p),(v,q),dΩ,dΓ) =
    ∫( ∇(v)⊙∇(u) - q*(∇⋅u) - (∇⋅v)*p ) * dΩ +
    ∫( (γd/hd)*v⋅u - v⋅(nΓ⋅∇(u)) - (nΓ⋅∇(v))⋅u + (p*nΓ)⋅v + (q*nΓ)⋅u ) * dΓ

  l((v,q),dΩ) = ∫( f⋅v ) * dΩ

  res(u,v,dΩ,dΓ) = a(u,v,dΩ,dΓ) - l(v,dΩ)

  domains = FEDomains((Ω,Γ),(Ω,Γ))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact_u = FESpace(Ωact,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  testagg_u = AgFEMSpace(testact_u,aggregates)
  testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
  testagg_p = AgFEMSpace(testact_p,aggregates)

  test_u = DirectSumFESpace(testbg_u,testagg_u)
  trial_u = TrialFESpace(test_u,g)
  test_p = DirectSumFESpace(testbg_p,testagg_p)
  trial_p = test_p
  test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
  trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

  ExtensionLinearOperator(res,a,trial,test,domains)
end

function get_trians(μ)
  x0 = Point(μ[1],μ[1]) #Point(1.0,1.0)#
  geo = !disk(0.35,x0=x0) #!disk(μ[2],x0=x0) #
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Ωout = Triangulation(cutgeo,PHYSICAL_OUT)
  Γ = EmbeddedBoundary(cutgeo)

  return Ω,Ωout,Γ
end

function compute_err(x,x̂,μ)
  x0 = Point(μ[1],μ[1]) #Point(1.0,1.0)#
  geo = !disk(0.35,x0=x0) #!disk(μ[2],x0=x0) #
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)
  nΓ = get_normal_vector(Γ)

  order = 2
  degree = 2*order

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact_u = FESpace(Ωact,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
  testagg_u = AgFEMSpace(testact_u,aggregates)
  testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
  testagg_p = AgFEMSpace(testact_p,aggregates)

  energy_u(du,v) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ
  energy_p(dp,q) = ∫(dp*q)dΩ
  Xu = assemble_matrix(energy_u,testagg_u,testagg_u)
  Xp = assemble_matrix(energy_p,testagg_p,testagg_p)

  d2bgd_u = Extensions.get_fdof_to_bg_fdof(testbg_u,testagg_u)
  d2bgd_p = Extensions.get_fdof_to_bg_fdof(testbg_p,testagg_p)

  ud = blocks(x)[1][d2bgd_u]
  ûd = blocks(x̂)[1][d2bgd_u]
  pd = blocks(x)[2][d2bgd_p]
  p̂d = blocks(x̂)[2][d2bgd_p]

  errH1u = compute_relative_error(ud,ûd,Xu)
  errL2p = compute_relative_error(pd,p̂d,Xp)
  return (errH1u,errL2p)
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
  uu,pp = blocks(x)
  ûû,p̂p̂ = blocks(x̂)

  if method==:ttsvd
    u,û,eu = vec(uu[:,:,:,i]),vec(ûû[:,:,:,i]),abs.(vec(uu[:,:,:,i]-ûû[:,:,:,i]))
    p,p̂,ep = vec(pp[:,:,i]),vec(p̂p̂[:,:,i]),abs.(vec(pp[:,:,i]-p̂p̂[:,:,i]))
  else
    u,û,eu = uu[:,i],ûû[:,i],abs.(uu[:,i]-ûû[:,i])
    p,p̂,ep = pp[:,i],p̂p̂[:,i],abs.(pp[:,i]-p̂p̂[:,i])
  end
  uh = FEFunction(trialbg_u,u)
  ûh = FEFunction(trialbg_u,û)
  euh = FEFunction(testbg_u,eu)
  ph = FEFunction(trialbg_p,p)
  p̂h = FEFunction(trialbg_p,p̂)
  eph = FEFunction(testbg_p,ep)

  Ω, = get_trians(μon.params[i])

  cellfields = ["uh"=>uh,"ûh"=>ûh,"euh"=>euh,"ph"=>ph,"p̂h"=>p̂h,"eph"=>eph]
  writevtk(Ω,joinpath(dir,"trian"),cellfields=cellfields)
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

  uu,pp = blocks(fesnaps)
  ûû,p̂p̂ = blocks(rbsnaps)

  error = [0.0,0.0]
  for (i,μ) in enumerate(μon)
    if method==:ttsvd
      u,û = vec(uu[:,:,:,i]),vec(ûû[:,:,:,i])
      p,p̂ = vec(pp[:,:,i]),vec(p̂p̂[:,:,i])
    else
      u,û = uu[:,i],ûû[:,i]
      p,p̂ = pp[:,i],p̂p̂[:,i]
    end
    xi = mortar([u,p])
    x̂i = mortar([û,p̂])
    error .+= compute_err(xi,x̂i,μ)
  end
  error ./= param_length(μon)

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
max_subspace_size(r::MultiFieldRBSpace) = sum(map(max_subspace_size,r))
max_subspace_size(a::Projection) = num_reduced_dofs(a)
max_subspace_size(a::Union{BlockProjection,BlockHRProjection}) = sum(map(max_subspace_size,a.array))
max_subspace_size(a::NormedProjection) = max_subspace_size(a.projection)
max_subspace_size(a::AffineContribution) = maximum(map(max_subspace_size,a.values))

function max_subspace_size(a::LocalProjection)
  maxsize = 0
  for proj in a.projections
    maxsize = max(maxsize,max_subspace_size(proj))
  end
  return maxsize
end

test_dir = datadir("moving_stokes")
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
fesnaps, = solution_snapshots(rbsolver,feop)
x,festats = solution_snapshots(rbsolver,feopon,μon)

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

perfs = ROMPerformance[]
maxsizes = Vector{Int}[]

for tol in (1e-0,1e-1,1e-2,1e-3,1e-4)
  tolrank = tol_or_rank(tol,rank)
  tolrank = method == :ttsvd ? fill(tolrank,3) : tolrank
  rbsolver = rb_solver(tolrank,nparams)

  dir = joinpath(test_dir,string(tol))
  create_dir(dir)

  rbop = reduced_operator(rbsolver,feop,fesnaps,jacs,ress)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  perf = postprocess(dir,rbsolver,feopon,rbop,x,x̂,festats,rbstats)
  maxsize = max_subspace_size(rbop)

  println(μon[1])
  println(perf)
  push!(perfs,perf)
  push!(maxsizes,maxsize)
end

serialize(joinpath(test_dir,"resultsH1"),perfs)
serialize(joinpath(test_dir,"maxsizes"),maxsizes)

end
