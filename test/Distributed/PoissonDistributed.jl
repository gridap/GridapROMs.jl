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

end # module

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

parts = (2,2)
snp = Ref{DistributedSnapshots}()
nrm = Ref{PSparseMatrix}()
basis = Ref{AbstractMatrix}()
feop = Ref{GenericParamOperator}()
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
  # snp[] = snaps
  nrm[] = X
  # # U,_,_ = tpod(LRApproxRank(1e-4),snaps,X)
  # # basis[] = U
  feop[] = op
  red_trial,red_test = reduced_spaces(red,op,snaps)
  jacs = jacobian_snapshots(rbsolver,op,snaps)
  ress = residual_snapshots(rbsolver,op,snaps)
  # red_jac = reduced_jacobian(jac_red,red_trial,red_test,jacs)
  red_res = reduced_residual(res_red,red_test,ress)
  # snp[] = jacs
end
