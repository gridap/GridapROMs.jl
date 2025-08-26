module PoissonDistributed

using Gridap
using GridapROMs
using Gridap.Algebra
using Gridap.FESpaces
using GridapDistributed
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using PartitionedArrays
using Test

function main(distribute, parts)
  ranks = distribute(LinearIndices((prod(parts),)))
  output = mkpath(joinpath(@__DIR__,"output"))

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

  μ = realization(pspace;nparams=10,sampling=:uniform)
  sol, = solve(LUSolver(),op,μ)

  μi = first(μ)
  Ui = TrialFESpace(V,u(μi))
  ai(u,v) = ∫( ∇(v)⋅∇(u) )dΩ
  li(v) = ∫( v*f(μi) )dΩ + ∫( v*(n_Γn⋅∇(u(μi))) )dΓn
  opi = AffineFEOperator(ai,li,Ui,V)
  uh = solve(opi)

  uhi = FEFunction(Ui,param_getindex(sol,1))

  eh = uh - uhi
  @test sqrt(sum( ∫(abs2(eh))dΩ )) < 1.0e-9

  writevtk(Ω,joinpath(output,"poisson"),cellfields=["uh" => uh,"uhμ" => uhi,"eh" => eh])
end

# with_debug() do distribute
#   main(distribute,(2,2))
# end

with_mpi() do distribute
  main(distribute,(2,2))
end

end # module
