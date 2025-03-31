module MeshTests

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.MultiField
using Test
using BenchmarkTools
using DrWatson

using ROManifolds
import ROManifolds.Utils: CostTracker
import ROManifolds.ParamDataStructures: GenericSnapshots,BlockSnapshots,TransientSnapshotsWithIC,get_realization,get_param_data

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
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
  )

  pdomain = (1,10,-1,5,1,2)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (n,n)

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

  energy(du,v,dΩ) = ∫(v*du)dΩ #+ ∫(∇(v)⋅∇(du))dΩ

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
  reduced_spaces(rbsolver,feop,fesnaps)

  println("pod no randomized")
  rbsolver = RBSolver(fesolver,PODReduction(tolrank,(du,v)->energy(du,v,dΩ);nparams);nparams_res,nparams_jac)
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
  state_reduction = Reduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe,sketch)
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
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
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

  energy(du,v,dΩ) = ∫(v*du)dΩ #+ ∫(∇(v)⋅∇(du))dΩ

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
  state_reduction = Reduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe,sketch)
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  dof_map = get_dof_map(test)
  fesnaps′ = change_dof_map(fesnaps,dof_map)
  reduced_spaces(rbsolver,feop,fesnaps′)
  # rbop = reduced_operator(rbsolver,feop,fesnaps′)
  # dir = datadir("poisson3d/$(method)/$(n)")
  # create_dir(dir)
  # save(dir,rbop)
end

# function main_stokes_2d(n;
#   tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
#   nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
#   )

#   pdomain = (1,10,-1,5,1,2)
#   pspace = ParamSpace(pdomain)

#   domain = (0,1,0,1)
#   partition = (n,n)

#   order = 2
#   degree = 2*order

#   a(μ) = x -> μ[1]*exp(-x[1])
#   aμ(μ) = ParamFunction(a,μ)

#   g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0)*(x[1]==0.0)
#   gμ(μ) = ParamFunction(g,μ)

#   stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
#   res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

#   energy((du,dp),(v,q),dΩ) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

#   reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   fesolver = LUSolver()

#   method = :pod
#   println("$(method) test stokes 2d n = $n")

#   model = CartesianDiscreteModel(domain,partition)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)
#   trian_res = (Ω,)
#   trian_stiffness = (Ω,)
#   domains = FEDomains(trian_res,trian_stiffness)

#   test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
#   trial_u = ParamTrialFESpace(test_u,gμ)

#   test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
#   trial_p = ParamTrialFESpace(test_p)
#   test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

#   tolrank = tol_or_rank(tol,rank)
#   coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
#   state_reduction = SupremizerReduction(coupling,tolrank,(du,v)->energy(du,v,dΩ);nparams)
#   rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

#   fesnaps, = solution_snapshots(rbsolver,feop)
#   rbop = reduced_operator(rbsolver,feop,fesnaps)
#   dir = datadir("stokes2d/$(method)/$(n)")
#   create_dir(dir)
#   save(dir,rbop)

#   method = :ttsvd
#   println("$(method) test stokes 2d n = $n")

#   model = TProductDiscreteModel(domain,partition)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)
#   trian_res = (Ω,)
#   trian_stiffness = (Ω,)
#   domains = FEDomains(trian_res,trian_stiffness)

#   test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
#   trial_u = ParamTrialFESpace(test_u,gμ)
#   test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
#   trial_p = ParamTrialFESpace(test_p)
#   test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

#   tolranks = fill(tolrank,4)
#   ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ
#   state_reduction = SupremizerReduction(ttcoupling,tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe)
#   rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

#   dof_map = get_dof_map(test)
#   fesnaps′ = change_dof_map(fesnaps,dof_map)
#   rbop = reduced_operator(rbsolver,feop,fesnaps′)
#   dir = datadir("stokes2d/$(method)/$(n)")
#   create_dir(dir)
#   save(dir,rbop)
# end

# function main_stokes_3d(n;
#   tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
#   nparams_jac=floor(Int,nparams/4),sketch=:sprn,unsafe=false
#   )

#   pdomain = (1,10,-1,5,1,2)
#   pspace = ParamSpace(pdomain)

#   domain = (0,1,0,1,0,1)
#   partition = (n,n,n)

#   order = 2
#   degree = 2*order

#   a(μ) = x -> μ[1]*exp(-x[1])
#   aμ(μ) = ParamFunction(a,μ)

#   g(μ) = x -> VectorValue(-(μ[2]*x[2]+μ[3])*x[2]*(1.0-x[2]),0.0,0.0)*(x[1]==0.0)
#   gμ(μ) = ParamFunction(g,μ)

#   stiffness(μ,(u,p),(v,q),dΩ) = ∫(aμ(μ)*∇(v)⊙∇(u))dΩ - ∫(p*(∇⋅(v)))dΩ + ∫(q*(∇⋅(u)))dΩ
#   res(μ,(u,p),(v,q),dΩ) = stiffness(μ,(u,p),(v,q),dΩ)

#   energy((du,dp),(v,q),dΩ) = ∫(du⋅v)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫(dp*q)dΩ

#   fesolver = LUSolver()
#   reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)

#   method = :pod
#   println("$(method) test stokes 3d n = $n")

#   model = CartesianDiscreteModel(domain,partition)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)
#   trian_res = (Ω,)
#   trian_stiffness = (Ω,)
#   domains = FEDomains(trian_res,trian_stiffness)

#   test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
#   trial_u = ParamTrialFESpace(test_u,gμ)
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
#   trial_p = ParamTrialFESpace(test_p)
#   test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

#   tolrank = tol_or_rank(tol,rank)
#   coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ
#   state_reduction = SupremizerReduction(coupling,tolrank,(du,v)->energy(du,v,dΩ);nparams)
#   rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

#   fesnaps, = solution_snapshots(rbsolver,feop)
#   rbop = reduced_operator(rbsolver,feop,fesnaps)
#   dir = datadir("stokes3d/$(method)/$(n)")
#   create_dir(dir)
#   save(dir,rbop)

#   method = :ttsvd
#   println("$(method) test stokes 3d n = $n")

#   model = TProductDiscreteModel(domain,partition)
#   Ω = Triangulation(model)
#   dΩ = Measure(Ω,degree)
#   trian_res = (Ω,)
#   trian_stiffness = (Ω,)
#   domains = FEDomains(trian_res,trian_stiffness)

#   test_u = TestFESpace(Ω,reffe_u;conformity=:H1,dirichlet_tags=[1,3,5,7,13,15,17,19,25])
#   trial_u = ParamTrialFESpace(test_u,gμ)
#   reffe_p = ReferenceFE(lagrangian,Float64,order-1)
#   test_p = TestFESpace(Ω,reffe_p;conformity=:H1)
#   trial_p = ParamTrialFESpace(test_p)
#   test = MultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
#   trial = MultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
#   feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)

#   tolranks = fill(tolrank,5)
#   ttcoupling((du,dp),(v,q)) = ∫(dp*∂₁(v))dΩ + ∫(dp*∂₂(v))dΩ + ∫(dp*∂₃(v))dΩ
#   state_reduction = SupremizerReduction(ttcoupling,tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe)
#   rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

#   dof_map = get_dof_map(test)
#   fesnaps′ = change_dof_map(fesnaps,dof_map)
#   rbop = reduced_operator(rbsolver,feop,fesnaps′)
#   dir = datadir("stokes2d/$(method)/$(n)")
#   create_dir(dir)
#   save(dir,rbop)
# end
function main_heateq_2d(n;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn,unsafe=false
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
  state_reduction = TTSVDReduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe,sketch)
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
  nparams_jac=floor(Int,nparams/4),nparams_djac=1,sketch=:sprn,unsafe=false
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
  state_reduction = TTSVDReduction(tolranks,(du,v)->energy(du,v,dΩ);nparams,unsafe,sketch)
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

# for n in (2,20,50,100,200,500)
#   main_poisson_2d(n)
# end

# for n in (2,20,40,60)
#   main_poisson_3d(n)
# end

# for n in (2,80)
#   main_stokes_2d(n)
# end

# for n in (2,10,15,20,25)
#   main_stokes_3d(n)
# end

# for n in (2,20,40,60,80,100)
#   main_heateq_2d(n)
# end

for n in (2,20)
  main_heateq_3d(n)
end

# res_jac_2d()
# res_jac_3d()

end

# using Gridap
# using Gridap.MultiField
# using Test
# using DrWatson

# using ROManifolds
# pdomain = (1,10,-1,5,1,2)
# pspace = ParamSpace(pdomain)

# domain = (0,1,0,1)
# partition = (200,200)

# order = 1
# degree = 2*order

# model = CartesianDiscreteModel(domain,partition)
# Ω = Triangulation(model)
# dΩ = Measure(Ω,degree)
# Γn = BoundaryTriangulation(model,tags=[8])
# dΓn = Measure(Γn,degree)

# a(μ) = x -> μ[1]
# aμ(μ) = ParamFunction(a,μ)

# f(μ) = x -> μ[1]
# fμ(μ) = ParamFunction(f,μ)

# a(μ,u,v) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
# rhs(μ,v) = ∫(fμ(μ)*v)dΩ + ∫(fμ(μ)*v)dΓn
# b(μ,u,v) = a(μ,u,v) - rhs(μ,v)

# reffe = ReferenceFE(lagrangian,Float64,order)
# test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
# trial = ParamTrialFESpace(test,fμ)
# feop = LinearParamOperator(b,a,pspace,trial,test)
# assem = SparseMatrixAssembler(trial,test)

# for nparams in (20,50,100,200,500)
#   μ = realization(pspace;nparams)
#   assemμ = parameterize(assem,μ)
#   feopμ = parameterize(feop,μ)
#   U = trial(μ)
#   x = zero_free_values(U)
#   tjac = @btimed A = allocate_jacobian(feopμ,x)
#   tres = @btimed b = allocate_residual(feopμ,x)
#   tjac! = @btimed jacobian!(A,feopμ,x)
#   tres! = @btimed residual!(b,feopμ,x)

#   ctjac = CostTracker(tjac,name="Allocation jacobian";nruns=nparams)
#   ctres = CostTracker(tjac,name="Allocation residual";nruns=nparams)
#   ctjac! = CostTracker(tjac,name="Assembly jacobian";nruns=nparams)
#   ctres! = CostTracker(tjac,name="Assembly residual";nruns=nparams)

#   println(ctjac)
#   println(ctres)
#   println(ctjac!)
#   println(ctres!)
# end
