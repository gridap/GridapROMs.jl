module MeshTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using LinearAlgebra
using SparseArrays
using Test
using BenchmarkTools
using DrWatson

using GridapROMs
import GridapROMs.Utils: CostTracker
import GridapROMs.ParamDataStructures: GenericSnapshots,BlockSnapshots,TransientSnapshotsWithIC,get_realization,get_param_data

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function change_dof_map(s::GenericSnapshots,dof_map)
  pdata = get_param_data(s)
  r = get_realization(s)
  Snapshots(pdata,dof_map,r)
end

function change_dof_map(s::TransientSnapshotsWithIC,dof_map)
  TransientSnapshotsWithIC(s.initial_data,change_dof_map(s.snaps,dof_map))
end

function change_dof_map(s::BlockSnapshots,dof_map::AbstractVector)
  N = ndims(s)
  array = Array{Snapshots,N}(undef,size(s))
  touched = s.touched
  for i in eachindex(touched)
    if touched[i]
      array[i] = change_dof_map(s[i],dof_map[i])
    end
  end
  return BlockSnapshots(array,touched)
end

function main_poisson_2d(n;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  pdomain = (1,5,1,5,1,5,1,5,1,5)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (n,n)

  order = 1
  degree = 2*order

  ν(μ) = x -> μ[1]*exp(-μ[2]*x[1])
  νμ(μ) = parameterize(ν,μ)

  f(μ) = x -> μ[3]
  fμ(μ) = parameterize(f,μ)

  g(μ) = x -> exp(-μ[4]*x[2])
  gμ(μ) = parameterize(g,μ)

  h(μ) = x -> μ[5]
  hμ(μ) = parameterize(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  energy(du,v,dΩ) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  fesolver = LUSolver()

  method = :pod
  println("$(method) test poisson 2d n = $n")

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  println("poisson 2d n = $n --------> Nh = $(num_free_dofs(test))")

  tolrank = tol_or_rank(tol,rank)
  state_reduction = PODReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  fesnaps, = solution_snapshots(rbsolver,feop)
  save(dir,fesnaps)
  reduced_spaces(rbsolver,feop,fesnaps)

  method = :ttsvd
  println("$(method) test poisson 2d n = $n")

  model = TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  tolranks = fill(tolrank,3)
  state_reduction = Reduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  dof_map = get_dof_map(test)
  fesnaps′ = change_dof_map(fesnaps,dof_map)
  reduced_spaces(rbsolver,feop,fesnaps′)
  # rbop = reduced_operator(rbsolver,feop,fesnaps′)
  # dir = datadir("poisson2d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)
end

function main_poisson_3d(n;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1,0,1)
  partition = (n,n,n)

  order = 1
  degree = 2*order

  a(μ) = x -> μ[1]
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = ParamFunction(f,μ)

  g(μ) = x -> μ[1]
  gμ(μ) = ParamFunction(g,μ)

  h(μ) = x -> μ[3]
  hμ(μ) = ParamFunction(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  energy(du,v,dΩ) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  fesolver = LUSolver()

  method = :pod
  println("$(method) test poisson 3d n = $n")

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  println("poisson 3d n = $n --------> Nh = $(num_free_dofs(test))")

  tolrank = tol_or_rank(tol,rank)
  state_reduction = PODReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  fesnaps, = solution_snapshots(rbsolver,feop)
  reduced_spaces(rbsolver,feop,fesnaps)
  # dir = datadir("poisson3d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)

  println("pod no randomized")
  rbsolver = RBSolver(fesolver,PODReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams);nparams_res,nparams_jac)
  reduced_spaces(rbsolver,feop,fesnaps)

  method = :ttsvd
  println("$(method) test poisson 3d n = $n")

  model = TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = ParamTrialFESpace(test,gμ)
  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

  tolranks = fill(tolrank,4)
  state_reduction = Reduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  dof_map = get_dof_map(test)
  fesnaps′ = change_dof_map(fesnaps,dof_map)
  reduced_spaces(rbsolver,feop,fesnaps′)
  # rbop = reduced_operator(rbsolver,feop,fesnaps′)
  # dir = datadir("poisson3d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)
end

function main_heateq_2d(n;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn
  )

  pdomain = (1,10,-1,5,1,2)
  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = n*dt
  tdomain = t0:dt:tf
  ptspace = TransientParamSpace(pdomain,tdomain)

  domain = (0,1,0,1)
  partition = (n,n)

  order = 1
  degree = 2*order

  a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
  aμt(μ,t) = TransientParamFunction(a,μ,t)

  f(μ,t) = x -> 1.
  fμt(μ,t) = TransientParamFunction(f,μ,t)

  h(μ,t) = x -> abs(cos(t/μ[3]))
  hμt(μ,t) = TransientParamFunction(h,μ,t)

  g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  u0(μ) = x -> 0.0
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

  energy(du,v,dΩ) = ∫(v*du)dΩ #+ ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  method = :pod
  println("$(method) test heateq 2d n = $n")

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  println("heateq 2d n = $n --------> Nh = $(num_free_dofs(test))")

  tolrank = tol_or_rank(tol,rank)
  state_reduction = TransientReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
  reduced_spaces(rbsolver,feop,fesnaps)
  # dir = datadir("heateq2d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)

  println("pod no randomized")
  rbsolver = RBSolver(fesolver,TransientReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams);nparams_res,nparams_jac,nparams_djac)
  reduced_spaces(rbsolver,feop,fesnaps)

  method = :ttsvd
  println("$(method) test heateq 2d n = $n")

  model = TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  tolranks = fill(tolrank,4)
  state_reduction = TTSVDReduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  dof_map = get_dof_map(test)
  fesnaps′ = change_dof_map(fesnaps,dof_map)
  reduced_spaces(rbsolver,feop,fesnaps′)
  # rbop = reduced_operator(rbsolver,feop,fesnaps′)
  # dir = datadir("heateq2d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)
end

function main_heateq_3d(n;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn
  )

  pdomain = (1,10,-1,5,1,2)
  θ = 0.5
  dt = 0.01
  t0 = 0.0
  tf = n*dt
  tdomain = t0:dt:tf
  ptspace = TransientParamSpace(pdomain,tdomain)

  domain = (0,1,0,1,0,1)
  partition = (n,n,n)

  order = 1
  degree = 2*order

  a(μ,t) = x -> 1+exp(-sin(t)^2*x[1]/sum(μ))
  aμt(μ,t) = TransientParamFunction(a,μ,t)

  f(μ,t) = x -> 1.
  fμt(μ,t) = TransientParamFunction(f,μ,t)

  h(μ,t) = x -> abs(cos(t/μ[3]))
  hμt(μ,t) = TransientParamFunction(h,μ,t)

  g(μ,t) = x -> μ[1]*exp(-x[1]/μ[2])*abs(sin(t/μ[3]))
  gμt(μ,t) = TransientParamFunction(g,μ,t)

  u0(μ) = x -> 0.0
  u0μ(μ) = ParamFunction(u0,μ)

  stiffness(μ,t,u,v,dΩ) = ∫(aμt(μ,t)*∇(v)⋅∇(u))dΩ
  mass(μ,t,uₜ,v,dΩ) = ∫(v*uₜ)dΩ
  rhs(μ,t,v,dΩ,dΓn) = ∫(fμt(μ,t)*v)dΩ + ∫(hμt(μ,t)*v)dΓn
  res(μ,t,u,v,dΩ,dΓn) = mass(μ,t,∂t(u),v,dΩ) + stiffness(μ,t,u,v,dΩ) - rhs(μ,t,v,dΩ,dΓn)

  energy(du,v,dΩ) = ∫(v*du)dΩ #+ ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)

  fesolver = ThetaMethod(LUSolver(),dt,θ)

  method = :pod
  println("$(method) test heateq 3d n = $n")

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  println("heateq 3d n = $n --------> Nh = $(num_free_dofs(test))")

  tolrank = tol_or_rank(tol,rank)
  state_reduction = TransientReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  fesnaps, = solution_snapshots(rbsolver,feop,uh0μ)
  reduced_spaces(rbsolver,feop,fesnaps)
  # dir = datadir("heateq3d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)

  println("pod no randomized")
  rbsolver = RBSolver(fesolver,TransientReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams);nparams_res,nparams_jac,nparams_djac)
  reduced_spaces(rbsolver,feop,fesnaps)

  method = :ttsvd
  println("$(method) test heateq 3d n = $n")

  model = TProductDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)
  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  trian_mass = (Ω,)
  domains = FEDomains(trian_res,(trian_stiffness,trian_mass))

  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = TransientTrialParamFESpace(test,gμt)
  feop = TransientParamLinearOperator((stiffness,mass),res,ptspace,trial,test,domains)
  uh0μ(μ) = interpolate_everywhere(u0μ(μ),trial(μ,t0))

  tolranks = fill(tolrank,4)
  state_reduction = TTSVDReduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,nparams_djac)

  dof_map = get_dof_map(test)
  fesnaps′ = change_dof_map(fesnaps,dof_map)
  reduced_spaces(rbsolver,feop,fesnaps′)
  # dir = datadir("heateq3d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)
end

function res_jac_2d()
  println("Running 2d assembly tests")
  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (200,200)

  order = 1
  degree = 2*order

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> μ[1]
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> μ[1]
  fμ(μ) = ParamFunction(f,μ)

  a(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v) = ∫(fμ(μ)*v)dΩ + ∫(fμ(μ)*v)dΓn
  l(μ,u,v) = a(μ,u,v) - rhs(μ,v)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,fμ)
  feop = LinearParamOperator(l,a,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)

  for nparams in (1,)
    μ = realization(pspace;nparams)
    assemμ = parameterize(assem,μ)
    feopμ = parameterize(feop,μ)
    U = trial(μ)
    x = zero_free_values(U)
    tjac = @btimed allocate_jacobian($feopμ,$x)
    tres = @btimed allocate_residual($feopμ,$x)
    A = tjac.value
    b = tres.value
    tjac! = @btimed jacobian!($A,$feopμ,$x)
    tres! = @btimed residual!($b,$feopμ,$x)

    ctjac = CostTracker(tjac,name="Allocation jacobian";nruns=nparams)
    ctres = CostTracker(tres,name="Allocation residual";nruns=nparams)
    ctjac! = CostTracker(tjac!,name="Assembly jacobian";nruns=nparams)
    ctres! = CostTracker(tres!,name="Assembly residual";nruns=nparams)

    println(ctjac)
    println(ctres)
    println(ctjac!)
    println(ctres!)
  end
end

function res_jac_3d()
  println("Running 3d assembly tests")
  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1,0,1)
  partition = (60,60,60)

  order = 1
  degree = 2*order

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[26])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> μ[1]
  aμ(μ) = ParamFunction(a,μ)

  f(μ) = x -> μ[1]
  fμ(μ) = ParamFunction(f,μ)

  a(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v) = ∫(fμ(μ)*v)dΩ + ∫(fμ(μ)*v)dΓn
  l(μ,u,v) = a(μ,u,v) - rhs(μ,v)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
  trial = ParamTrialFESpace(test,fμ)
  feop = LinearParamOperator(l,a,pspace,trial,test)
  assem = SparseMatrixAssembler(trial,test)

  for nparams in (1,)
    μ = realization(pspace;nparams)
    assemμ = parameterize(assem,μ)
    feopμ = parameterize(feop,μ)
    U = trial(μ)
    x = zero_free_values(U)
    tjac = @btimed allocate_jacobian($feopμ,$x)
    tres = @btimed allocate_residual($feopμ,$x)
    A = tjac.value
    b = tres.value
    tjac! = @btimed jacobian!($A,$feopμ,$x)
    tres! = @btimed residual!($b,$feopμ,$x)

    ctjac = CostTracker(tjac,name="Allocation jacobian";nruns=nparams)
    ctres = CostTracker(tres,name="Allocation residual";nruns=nparams)
    ctjac! = CostTracker(tjac!,name="Assembly jacobian";nruns=nparams)
    ctres! = CostTracker(tres!,name="Assembly residual";nruns=nparams)

    println(ctjac)
    println(ctres)
    println(ctjac!)
    println(ctres!)
  end
end

function d_cholesky(partition::NTuple{D}) where D
  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  if D == 1
    domain = (0,1)
  elseif D == 2
    domain = (0,1,0,1)
  elseif D == 3
    domain = (0,1,0,1,0,1)
  else
    error("not implemented")
  end

  order = 1
  degree = 2*order

  model = CartesianDiscreteModel(domain,partition)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1)
  trial = ParamTrialFESpace(test,x->x)
  feop = LinearParamOperator(x->x,x->x,pspace,trial,test)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  x = rand(num_free_dofs(test))

  println("---------------------------------")
  println("Running $D - dimensional cholesky test with Ns = $(length(x))")

  X = assemble_matrix(feop,energy)
  t1 = @btimed assemble_matrix($feop,$energy)
  X = t1.value
  t2 = @btimed cholesky($X)
  H = t2.value
  t3 = @btimed $H \ $x

  println(CostTracker(t1,name="X assembly, Nz(X) = $(nnz(X))"))
  println(CostTracker(t2,name="H = cholesky(X)"))
  println(CostTracker(t3,name="H div x"))
  println("---------------------------------")
end

end
