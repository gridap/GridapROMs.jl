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
f(μ) = x -> -Δ(sol(μ),x)
fμ(μ) = parameterize(f,μ)
solμ(μ) = parameterize(sol,μ)

pspace = ParamSpace((1,2))
μ = Realization([[1.0],[2.0]])

function test_solver(solver,op,dΩ)
  test = get_test(op)
  trial = get_trial(op)
  y = zero_free_values(trial(μ))
  nlop = parameterize(op,μ)
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
  @test sum(∫(eh*eh)*dΩ) < 1.0e-6

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

with_mpi() do distribute
  main(distribute,(2,2))
end

end # module
