module PoissonDistributed

using Gridap
using GridapROMs
using GridapPETSc
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using PartitionedArrays
using Test

sol(μ) = x -> μ[1]*x[1] + x[2]
f(μ) = x -> -Δ(sol(μ))(x)
fμ(μ) = parameterise(f,μ)
solμ(μ) = parameterise(sol,μ)

pspace = ParamSpace((1,2))
μ = Realisation([[1.0],[2.0]])

function test_solver(solver,op,dΩ)
  test = get_test(op)
  trial = get_trial(op)
  y = zero_free_values(trial(μ))
  nlop = parameterise(op,μ)
  syscache = allocate_systemcache(nlop,y)
  A = get_matrix(syscache)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,solver,nlop,syscache)

  μi = first(μ)
  xi = param_getindex(x,1)
  Ui = TrialFESpace(test,sol(μi))
  uhi = FEFunction(Ui,xi)
  uh = interpolate(sol(μi),Ui)

  eh = uh - uhi
  @test sqrt(sum(∫(eh*eh)*dΩ)) < 1.0e-6

  return x
end

function get_mesh(parts,np)
  Dc = length(np)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (8,8)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (8,8,8)
  end
  if prod(np) == 1
    model = CartesianDiscreteModel(domain,nc)
  else
    model = CartesianDiscreteModel(parts,np,domain,nc)
  end
  return model
end

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor -ksp_rtol 1e-8"
  GridapPETSc.with(args=split(options)) do
    model = get_mesh(ranks,parts)

    order  = 1
    degree = order*2 + 1
    reffe  = ReferenceFE(lagrangian,Float64,order)
    test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    trial = ParamTrialFESpace(test,solμ)

    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    a(μ,u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(μ,u,v) = a(μ,u,v) - ∫( v*fμ(μ) )dΩ
    op = LinearParamOperator(l,a,pspace,trial,test)

    solver = PETScLinearSolver()

    test_solver(solver,op,dΩ)
  end
end

end

using Gridap
using GridapROMs
using GridapPETSc
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using GridapROMs.RBSteady
using GridapROMs.ParamSteady
using PartitionedArrays
using Test

sol(μ) = x -> μ[1]*x[1] + x[2]
f(μ) = x -> -Δ(sol(μ))(x)
fμ(μ) = parameterise(f,μ)
solμ(μ) = parameterise(sol,μ)

pspace = ParamSpace((1,2))
μ = Realisation([[1.0],[2.0]])

function test_solver(solver,op,dΩ)
  test = get_test(op)
  trial = get_trial(op)
  y = zero_free_values(trial(μ))
  nlop = parameterise(op,μ)
  syscache = allocate_systemcache(nlop,y)
  A = get_matrix(syscache)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,solver,nlop,syscache)

  μi = first(μ)
  xi = param_getindex(x,1)
  Ui = TrialFESpace(test,sol(μi))
  uhi = FEFunction(Ui,xi)
  uh = interpolate(sol(μi),Ui)

  eh = uh - uhi
  @test sqrt(sum(∫(eh*eh)*dΩ)) < 1.0e-6

  return x
end

function get_mesh(parts,np)
  Dc = length(np)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (8,8)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (8,8,8)
  end
  if prod(np) == 1
    model = CartesianDiscreteModel(domain,nc)
  else
    model = CartesianDiscreteModel(parts,np,domain,nc)
  end
  return model
end

red = PODReduction(1e-4,H1(),nparams=2)
rbsolver = RBSolver(LUSolver(),red,nparams_jac=2,nparams_res=2)
res_red = rbsolver.residual_reduction
jac_red = rbsolver.jacobian_reduction

with_debug() do distribute
  ranks = distribute(LinearIndices((prod(parts),)))

  model = get_mesh(ranks,parts)

  order  = 1
  degree = order*2 + 1
  reffe  = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  trial = ParamTrialFESpace(test,solμ)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  a(μ,u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(μ,u,v) = a(μ,u,v) - ∫( v*fμ(μ) )dΩ
  op = LinearParamOperator(l,a,pspace,trial,test)
  X = assemble_matrix((u,v) -> ∫( ∇(v)⋅∇(u) )dΩ,trial,test)

  y = zero_free_values(trial(μ))
  nlop = parameterise(op,μ)
  syscache = allocate_systemcache(nlop,y)
  A = get_matrix(syscache)
  b = get_vector(syscache)
  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,LUSolver(),nlop,syscache)
  snaps = Snapshots(x,get_dof_map(trial),μ)
  @test isa(snaps,DistributedSnapshots)
  red_trial,red_test = reduced_spaces(red,op,snaps)
  jacs = jacobian_snapshots(rbsolver,op,snaps)
  ress = residual_snapshots(rbsolver,op,snaps)
  red_jac = reduced_jacobian(jac_red,red_trial,red_test,jacs)
  red_res = reduced_residual(res_red,red_test,ress)
end

#

using Gridap
using GridapROMs
using GridapPETSc
using DrWatson
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using GridapROMs.Distributed
using PartitionedArrays
using Test

method=:pod
compression=:global
hypred_strategy=:deim
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:deim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :deim

domain = (0,1,0,1)
partition = (8,8)

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterise(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterise(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterise(h,μ)

order = 1
degree = 2*order

state_reduction = Reduction(tol,H1();nparams,sketch,compression,ncentroids)

snp = Ref{DistributedSnapshots}()
op = Ref{ReducedOperator}()

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor -ksp_rtol 1e-8"
  GridapPETSc.with(args=split(options)) do
    model = CartesianDiscreteModel(ranks,parts,domain,partition)
    
    Ω = Triangulation(model)
    dΩ = Measure(Ω,degree)
    Γn = BoundaryTriangulation(model,tags=[8])
    dΓn = Measure(Γn,degree)

    stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
    rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
    res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

    trian_res = (Ω,Γn)
    trian_stiffness = (Ω,)
    domains = FEDomains(trian_res,trian_stiffness)

    reffe = ReferenceFE(lagrangian,Float64,order)
    test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
    trial = ParamTrialFESpace(test,gμ)

    fesolver = LUSolver()#PETScLinearSolver()
    rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

    feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
    fesnaps, = solution_snapshots(rbsolver,feop)
    snp[] = fesnaps
    rbop = reduced_operator(rbsolver,feop,fesnaps)
    op[] = rbop

    μon = realisation(feop;nparams=10,sampling=:uniform)
    x̂,rbstats = solve(rbsolver,rbop,μon)
    x,festats = solution_snapshots(rbsolver,feop,μon)
    perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)
    println(perf)

    # ---------------------------------------------------------------------------
    # diagnostic test: separate the RB basis / projection error from the
    # hyper-reduction error.
    #
    #  * projection_error  : ‖s - Π s‖ / ‖s‖  (Π = project ∘ inv_project onto the
    #                        RB trial space). Uses NO hyper-reduction. If this is
    #                        ~tol the basis + project/inv_project are fine and the
    #                        online error comes from the (DEIM) hyper-reduction.
    #  * hr_error          : per-triangulation relative error of the hyper-reduced
    #                        residual / Jacobian vs. the FOM ones.
    # ---------------------------------------------------------------------------
    # never `show` the caught exception object directly: for the distributed case
    # its payload can hold a `DebugArray`, and `show` scalar-indexes it.
    _why(e) = sprint(showerror,e;context=(:limit=>true,:displaysize=>(4,120)))

    try
      perr = GridapROMs.RBSteady.projection_error(rbsolver,rbop,fesnaps)
      println("diagnostic | projection error (basis + project/inv_project, no HR): ", perr)
    catch e
      println("diagnostic | projection_error unavailable (distributed): ", first(split(_why(e),'\n')))
    end

    try
      res = residual_snapshots(rbsolver,feop,fesnaps)
      jac = jacobian_snapshots(rbsolver,feop,fesnaps)
      err_res,err_jac = GridapROMs.RBSteady.hr_error(rbsolver,rbop,res,jac,fesnaps)
      println("diagnostic | hr error residual (per trian): ", err_res)
      println("diagnostic | hr error jacobian (per trian): ", err_jac)
    catch e
      println("diagnostic | hr_error unavailable (distributed): ", first(split(_why(e),'\n')))
    end

    # per-rank save / load round-trip of the FE snapshots (distributed)
    diagdir = mkpath(joinpath(@__DIR__,"boh_diag"))
    save(diagdir,fesnaps)
    fesnaps_loaded = load_snapshots(diagdir,ranks)
    println("diagnostic | snapshots save/load round-trip ok: ",
      compute_relative_error(fesnaps,fesnaps_loaded) < 1e-12)
  end
end

with_debug() do distribute
  main(distribute,(2,2))
end

rbop = op[]
proj = rbop.test.subspace
ϕ = get_basis(proj.projection)
X = proj.norm_matrix
s = snp[]

ϕ'*(X*ϕ)

ϕ'*X*ϕ

using GridapROMs.Utils
using GridapROMs.ParamDataStructures
using GridapROMs.RBSteady
using BlockArrays
v = rand(3)
pv = parameterise(v,3)
bpv = mortar([pv,pv])
t = Triangulation(CartesianDiscreteModel((0,1,0,1),(8,8)))
c = Contribution((bpv,bpv),(t,t))
hr = HRParamArray(c,c,bpv)
fill!(c,0.0)