module PoissonDistributed

using Gridap
using GridapROMs
using GridapPETSc
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using PartitionedArrays
using Test

function main_ex1(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  domain = (0,4,0,4)
  cells = (4,4)
  model = CartesianDiscreteModel(ranks,parts,domain,cells)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
  add_tag_from_tags!(labels,"neumann",[4,6,8])

  Ω = Triangulation(model)
  Γn = Boundary(model,tags="neumann")
  n_Γn = get_normal_vector(Γn)

  k = 2
  u(μ) = x -> (μ[1]*x[1]+x[2])^k
  f(μ) = x -> -Δ(u(μ),x)
  fμ(μ) = parameterize(f,μ)
  uμ(μ) = parameterize(u,μ)

  reffe = ReferenceFE(lagrangian,Float64,k)
  V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
  U = ParamTrialFESpace(V,uμ)

  pspace = ParamSpace((0,1))

  dΩ = Measure(Ω,2*k)
  dΓn = Measure(Γn,2*k)

  a(μ,u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  l(μ,u,v) = a(μ,u,v) - ∫( v*fμ(μ) )dΩ - ∫( v*(n_Γn⋅∇(uμ(μ))) )dΓn
  op = LinearParamOperator(l,a,pspace,U,V)

  solver = LUSolver()

  μ = realization(pspace;nparams=10,sampling=:uniform)
  sol, = solve(solver,op,μ)

  μi = first(μ)
  Ui = TrialFESpace(V,u(μi))
  uhi = FEFunction(Ui,param_getindex(sol,1))

  eh = u(μi) - uhi
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
end

function main_ex2(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  options = "-ksp_type cg -pc_type gamg -ksp_monitor"
  GridapPETSc.with(args=split(options)) do
    domain = (0,4,0,4)
    cells = (4,4)
    model = CartesianDiscreteModel(ranks,parts,domain,cells)

    labels = get_face_labeling(model)
    add_tag_from_tags!(labels,"dirichlet",[1,2,3,5,7])
    add_tag_from_tags!(labels,"neumann",[4,6,8])

    Ω = Triangulation(model)
    Γn = Boundary(model,tags="neumann")
    n_Γn = get_normal_vector(Γn)

    k = 2
    u(μ) = x -> (μ[1]*x[1]+x[2])^k
    f(μ) = x -> -Δ(u(μ),x)
    fμ(μ) = parameterize(f,μ)
    uμ(μ) = parameterize(u,μ)

    reffe = ReferenceFE(lagrangian,Float64,k)
    V = TestFESpace(model,reffe,dirichlet_tags="dirichlet")
    U = ParamTrialFESpace(V,uμ)

    pspace = ParamSpace((0,1))

    dΩ = Measure(Ω,2*k)
    dΓn = Measure(Γn,2*k)

    a(μ,u,v) = ∫( ∇(v)⋅∇(u) )dΩ
    l(μ,u,v) = a(μ,u,v) - ∫( v*fμ(μ) )dΩ - ∫( v*(n_Γn⋅∇(uμ(μ))) )dΓn
    op = LinearParamOperator(l,a,pspace,U,V)

    solver = PETScLinearSolver()

    μ = realization(pspace;nparams=10,sampling=:uniform)
    sol, = solve(solver,op,μ)

    μi = first(μ)
    Ui = TrialFESpace(V,u(μi))
    uhi = FEFunction(Ui,param_getindex(sol,1))

    eh = u(μi) - uhi
    @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9
  end
end

with_mpi() do distribute
  main_ex1(distribute,(2,2))
end

with_mpi() do distribute
  main_ex2(distribute,(2,2))
end

end # module
