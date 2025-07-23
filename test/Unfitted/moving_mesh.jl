using DrWatson
using LinearAlgebra
using Serialization

using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.Arrays
using Gridap.Algebra
using Gridap.ReferenceFEs
using Gridap.FESpaces
using Gridap.ODEs
using GridapROMs.DofMaps
using GridapROMs.Uncommon
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.Utils

method=:pod
tol=1e-4
rank=nothing
nparams=100
nparams_res=nparams_jac=nparams

domain = (-1,1,-1,1)
n=40
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 1
degree = 2

pspace = ParamSpace((-0.5,0.5,0.1,0.3))

const γd = 10.0
const hd = 2/n

function get_deformation_map(μ)
  μ0 = (0.0,0.2)
  x0 = Point(μ0[1],μ0[1])
  geo = !disk(μ0[2],x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)
  φ(μ) = x -> VectorValue((μ[1]-μ0[1]) + (μ[2]/μ0[2])*x[1], (μ[1]-μ0[1]) + (μ[2]/μ0[2])*x[2])
  φμ(μ) = parameterize(φ,μ)

  λ = 0.33/((1+0.33)*(1-2*0.33))
  p = 1/(2(1+0.33))
  σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

  a(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⊙u )dΓ
  l(μ,v) = ∫( (γd/hd)*v⋅φμ(μ) - (n_Γ⋅(σ∘ε(v)))⋅φμ(μ) )dΓ
  res(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(μ,v)

  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  φ, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),φ)
end

φ = get_deformation_map(realization(pspace))

reffe = ReferenceFE(lagrangian,Float64,order)

Ω = Triangulation(bgmodel)
dΩ = Measure(Ω,degree)
energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

function def_fe_operator(μ)
  mmodel = MappedDiscreteModel(model,ϕμ(μ))

  Ωm = Triangulation(mmodel)
  Γm = BoundaryTriangulation(mmodel,tags=8)

  dΩm = Measure(Ωm,degree)
  dΓm = Measure(Γm,degree)

  Vm = TestFESpace(mmodel,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  Um = ParamTrialFESpace(Vm,gμ)

  am(μ,u,v,dΩm) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩm
  bm(μ,u,v,dΩm,dΓm) = am(μ,u,v,dΩm) - ( ∫(fμ(μ)*v)dΩm + ∫(fμ(μ)*v)dΓm )

  domains = FEDomains((Ωm,Γm),(Ωm,))

  LinearParamOperator(bm,am,pspace,Um,Vm,domains)
end

μ = realization(pspace;nparams)
feop = def_fe_operator(μ)

μon = realization(pspace;nparams,sampling=:uniform)
feopon = def_fe_operator(μon)

fesolver = LUSolver()
state_reduction = Reduction(tol;nparams,sketch=:sprn) #energy
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

fesnaps, = solution_snapshots(rbsolver,feop,μ)
rbop′ = reduced_operator(rbsolver,feop,fesnaps)
rbop = RBSteady.change_operator(rbop′,feopon)

x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feopon,μon)
perf = eval_performance(rbsolver,feopon,rbop,x,x̂,festats,rbstats)

function mdeim_error(op::RBOperator,r::AbstractRealization,jx,rx)
  jxdata = recast(jx)
  rxdata = get_all_data(rx)

  ĵ = galerkin_projection(get_basis(op.trial),jxdata,get_basis(op.test))
  r̂ = galerkin_projection(get_basis(op.test),rxdata)

  x̂ = zero_free_values(get_trial(op)(r))
  nlop = parameterize(op,r)
  syscache = allocate_systemcache(nlop,x̂)
  jacobian!(syscache.A,nlop,x̂)
  residual!(syscache.b,nlop,x̂)

  err_j,err_r = 0.0,0.0
  for i in param_eachindex(r)
    err_j += norm(ĵ[:,i,:]-syscache.A[i,i]) / norm(ĵ[:,i,:])
    err_r += norm(r̂[:,i]+syscache.b[i]) / norm(r̂[:,i])
  end

  return err_j/param_length(r),err_r/param_length(r)
end

jx′ = jacobian_snapshots(rbsolver,feopon,x)
jx = jx′[1]
rx′ = residual_snapshots(rbsolver,feopon,x)
rx = Snapshots(sum(rx′),get_dof_map(rx′[1]),get_realization(rx′[1]))
RBSteady.projection_error(rbop.trial,get_param_data(x),μon)
mdeim_error(rbop,μon,jx,rx)

μ = first(μon)
modelμ = MappedDiscreteModel(model,ϕ(μ))
Ωμ = Triangulation(modelμ)
dΩμ = Measure(Ωμ,degree)
Vμ = TestFESpace(modelμ,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
Uμ = TrialFESpace(Vμ,g(μ))
bμ = assemble_vector(v -> ∫(ν(μ)*∇(v)⋅∇(zero(Uμ)))dΩμ - ∫(f(μ)*v)dΩμ,Vμ)
writevtk(Ωμ,datadir("plts/mapped"),cellfields=["uh"=>FEFunction(Uμ,x[:,1])])

nodesμ = get_node_coordinates(modelμ)

jxdata = recast(jx)
rxdata = get_all_data(rx)

jrb = galerkin_projection(get_basis(rbop.trial),jxdata,get_basis(rbop.test))[:,1,:]
rrb = galerkin_projection(get_basis(rbop.test),rxdata)[:,1]

xrb = project(rbop.test,x[:,1])
yrb = jrb \ rrb
