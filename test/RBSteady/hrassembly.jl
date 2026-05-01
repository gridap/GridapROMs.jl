module HRAssemblyTests

using Test
using LinearAlgebra
using Gridap
using Gridap.FESpaces
using Gridap.Algebra
using GridapROMs
using GridapROMs.RBSteady

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
)
  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt) ? hypred_strategy : :mdeim

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (20,20)
  if method == :ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 1
  degree = 2 * order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1] / sum(μ))
  aμ(μ) = parameterise(a,μ)

  f(μ) = x -> 1.0
  fμ(μ) = parameterise(f,μ)

  g(μ) = x -> μ[1] * exp(-x[1] / μ[2])
  gμ(μ) = parameterise(g,μ)

  h(μ) = x -> abs(cos(μ[3] * x[2]))
  hμ(μ) = parameterise(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ) * ∇(v) ⋅ ∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ) * v)dΩ + ∫(hμ(μ) * v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v * du)dΩ + ∫(∇(v) ⋅ ∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  if method == :pod
    state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  else
    state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)

  ress = residual_snapshots(rbsolver,feop,fesnaps)
  res_snaps = ress[2]

  μ = res_snaps.realisation
  v = get_fe_basis(test)
  dc = ∫(hμ(μ) * v)dΓn
  vecdata = collect_cell_vector(test,dc)
  assem = SparseMatrixAssembler(trial(μ),test)
  b = allocate_vector(assem,vecdata)
  assemble_vector_add!(b,assem,vecdata)
  @test b.data ≈ -res_snaps

  red = rbsolver.residual_reduction
  rred = red.reduction
  Φres = projection(rred,res_snaps)
  proj_basis = project(red_test,Φres)
  @test get_basis(proj_basis) ≈ get_basis(red_test)' * get_basis(Φres)

  interp = Interpolation(red,Φres,Γn,red_test)
  r̂ = HRProjection(proj_basis,red,interp)

  trian = Γn
  rows,ΦresI = empirical_interpolation(Φres)
  cell_row_ids = get_cell_dof_ids(red_test,trian)
  cells = RBSteady.get_rows_to_cells(cell_row_ids,rows)
  irows = RBSteady.get_cells_to_idofs(cell_row_ids,cells,rows)

  μnew = realisation(pspace;sampling=:uniform)

  Û = red_trial(μnew)
  x̂ = zero_free_values(Û)
  b̂ = RBSteady.allocate_hypred_cache(r̂,μnew)
  fill!(b̂,zero(eltype(b̂)))

  v = get_fe_basis(red_test)

  strian = view(trian,cells)
  dcstrian = ∫(hμ(μnew) * v) * Measure(strian,degree)
  rhs_strian = RBSteady.get_interpolation(r̂)
  vecdata = RBSteady.collect_cell_hr_vector(test,dcstrian,strian,rhs_strian)
  assemble_hr_vector_add!(b̂.fecache,vecdata...)

  interpolate!(b̂.hypred,b̂.coeff,r̂,b̂.fecache)
  @test b̂.coeff.data ≈ ΦresI \ b̂.fecache.data
  @test b̂.hypred.data ≈ get_basis(proj_basis) * (ΦresI \ b̂.fecache.data)

  dc = ∫(hμ(μnew) * v)dΓn
  vecdata = collect_cell_vector(test,dc)
  assem = SparseMatrixAssembler(trial(μnew),test)
  bfe = allocate_vector(assem,vecdata)
  assemble_vector_add!(bfe,assem,vecdata)
  bfe_rows = bfe.data[rows,:]
  bred = get_basis(red_test)' * bfe.data

  @test length(cells) == length(irows)
  @test b̂.fecache.data ≈ bfe_rows
  @test b̂.hypred.data ≈ bred atol=2e-4 rtol=1e-6
end

@testset "HR assembly" begin
  @testset "method=pod" begin
    main(:pod)
  end
end

end # module