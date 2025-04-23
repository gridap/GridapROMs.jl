module PoissonEmbedded

using Gridap
using GridapEmbedded
using GridapROMs

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

function main(
  method=:pod,n=20;
  tol=1e-4,rank=nothing,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn
  )

  @assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  R = 0.5
  L = 0.8*(2*R)
  p1 = Point(0.0,0.0)
  p2 = p1 + VectorValue(L,0.0)

  geo1 = disk(R,x0=p1)
  geo2 = disk(R,x0=p2)
  geo = setdiff(geo1,geo2)

  t = 1.01
  pmin = p1-t*R
  pmax = p1+t*R
  dp = pmax - pmin

  partition = (n,n)
  if method==:ttsvd
    bgmodel = TProductDiscreteModel(pmin,pmax,partition)
  else
    bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
  end

  order = 2
  degree = 2*order

  cutgeo = cut(bgmodel,geo)
  Ωbg = Triangulation(bgmodel)
  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩbg = Measure(Ωbg,degree)
  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  nΓ = get_normal_vector(Γ)

  const γd = 10.0
  const hd = dp[1]/n

  ν(μ) = x->μ[3]
  νμ(μ) = ParamFunction(ν,μ)

  f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
  fμ(μ) = ParamFunction(f,μ)

  h(μ) = x->1
  hμ(μ) = ParamFunction(h,μ)

  g(μ) = x->μ[3]*x[1]-x[2]
  gμ(μ) = ParamFunction(g,μ)

  a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
  res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

  trian_a = (Ω,Γ)
  trian_res = (Ω,Γ,Γn)
  domains = FEDomains(trian_res,trian_a)

  reffe = ReferenceFE(lagrangian,Float64,order)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testbg = FESpace(Ωbg,reffe,conformity=:H1)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = ParamTrialFESpace(test,gμ)
  feop = ExtensionLinearParamOperator(res,a,pspace,trial,test,domains)

  energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
  tolrank = tol_or_rank(tol,rank)
  state_reduction = Reduction(tolrank,energy;nparams,sketch)

  trial = ParamTrialFESpace(test)
  feop = LinearParamOperator(b,a,pspace,trial,test,domains)

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

  # offline
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  # online
  μon = realization(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)

  # test
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
  println(perf)
end

main(:pod)
main(:ttsvd)

end

using Gridap
using Gridap.Algebra
using Gridap.Arrays
using Gridap.CellData
using Gridap.FESpaces
using GridapEmbedded
using GridapROMs
using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.Extensions
using GridapROMs.DofMaps
using GridapROMs.RBSteady
using GridapROMs.Utils
using SparseArrays
using DrWatson
using Test
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.BlockSolvers

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(1,1)
n = 20
partition = (n,n)

dp = pmax - pmin

geo1 = disk(R,x0=Point(0.5,0.5))
geo2 = ! geo1

model = TProductDiscreteModel(pmin,pmax,partition)
cutgeo = cut(model,geo2)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

Ωbg = Triangulation(model)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL_IN)
Γ = EmbeddedBoundary(cutgeo)
n_Γ = get_normal_vector(Γ)

order = 2
degree = 2*order

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)
dΓ = Measure(Γ,degree)

reffe = ReferenceFE(lagrangian,Float64,order)

V = FESpace(Ωbg,reffe,conformity=:H1)
Vact = FESpace(Ωact,reffe,conformity=:H1)
Vagg = AgFEMSpace(V,Vact,aggregates)

const γd = 10.0
const hd = dp[1]/n

ν(μ) = x->μ[3]
νμ(μ) = ParamFunction(ν,μ)

f(μ) = x->μ[1]*x[1] - μ[2]*x[2]
fμ(μ) = ParamFunction(f,μ)

h(μ) = x->1
hμ(μ) = ParamFunction(h,μ)

g(μ) = x->μ[3]*x[1]-x[2]
gμ(μ) = ParamFunction(g,μ)

a(μ,u,v,dΩ,dΓ) = ∫(νμ(μ)*∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(μ,v,dΩ,dΓ,dΓn) = ∫(fμ(μ)⋅v)dΩ + ∫(hμ(μ)⋅v)dΓn + ∫( (γd/hd)*v*gμ(μ) - (n_Γ⋅∇(v))*gμ(μ) )dΓ
res(μ,u,v,dΩ,dΓ,dΓn) =  a(μ,u,v,dΩ,dΓ) - l(μ,v,dΩ,dΓ,dΓn)

trian_a = (Ω,Γ)
trian_res = (Ω,Γ,Γn)
domains = FEDomains(trian_res,trian_a)

Vext = DirectSumFESpace(V,Vagg)
Uext = ParamTrialFESpace(Vext,gμ)

feop = ExtensionLinearParamOperator(res,a,pspace,Uext,Vext,domains)

solver = ExtensionSolver(LUSolver())
energy(u,v) = ∫(v*u)dΩbg + ∫(∇(v)⋅∇(u))dΩbg
state_reduction = Reduction(fill(1e-4,2),energy;nparams=50)
rbsolver = RBSolver(solver,state_reduction;nparams_res=50,nparams_jac=20)

fesnaps, = solution_snapshots(rbsolver,feop)

rbop = reduced_operator(rbsolver,feop,fesnaps)
μon = realization(feop;nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
jacs = jacobian_snapshots(rbsolver,feop,fesnaps)

bgcell_to_bgcellin = aggregates
trian_a = get_triangulation(Vact)
shfns_g = get_fe_basis(Vact)
dofs_g = get_fe_dof_basis(Vact)
bgcell_to_gcell = 1:length(bgcell_to_bgcellin)
bg_cell_dofs = get_cell_dof_ids(V)

D = num_cell_dims(trian_a)
glue = get_glue(trian_a,Val(D))
acell_to_bgcell = glue.tface_to_mface
bgcell_to_acell = glue.mface_to_tface
acell_to_bgcellin = collect(lazy_map(Reindex(bgcell_to_bgcellin),acell_to_bgcell))
acell_to_acellin = collect(lazy_map(Reindex(bgcell_to_acell),acell_to_bgcellin))
acell_to_gcell = lazy_map(Reindex(bgcell_to_gcell),acell_to_bgcell)

acell_phys_shapefuns_g = get_array(change_domain(shfns_g,PhysicalDomain()))
acell_phys_root_shapefuns_g = lazy_map(Reindex(acell_phys_shapefuns_g),acell_to_acellin)
root_shfns_g = GenericCellField(acell_phys_root_shapefuns_g,trian_a,PhysicalDomain())

dofs_f = get_fe_dof_basis(Vact)
shfns_f = get_fe_basis(Vact)
acell_to_coeffs = dofs_f(root_shfns_g)
acell_to_proj = dofs_g(shfns_f)
acell_to_dof_ids = get_cell_dof_ids(Vact)

acell_to_terms = DofMaps.get_term_to_bg_terms(V,Vact)

aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,fdof_to_terms = DofMaps._setup_oagfem_constraints(
  num_free_dofs(Vact),
  acell_to_acellin,
  acell_to_terms,
  acell_to_dof_ids,
  acell_to_coeffs,
  acell_to_proj,
  acell_to_gcell)

n_fdofs = num_free_dofs(Vact)
n_ddofs = num_dirichlet_dofs(Vact)
n_DOFs = n_fdofs+n_ddofs

(sDOF_to_dof,
sDOF_to_dofs,
sDOF_to_coeffs,
DOF_to_term,
sDOF_to_terms) = aggdof_to_fdof,aggdof_to_dofs,aggdof_to_coeffs,fdof_to_term,aggdof_to_terms

DOF_to_DOFs,DOF_to_coeffs,DOF_to_terms = DofMaps._prepare_oDOF_to_oDOFs(
  sDOF_to_dof,
  sDOF_to_dofs,
  sDOF_to_coeffs,
  DOF_to_term,
  sDOF_to_terms,
  n_fdofs,
  n_DOFs)

mDOF_to_DOF,n_fmdofs = FESpaces._find_master_dofs(DOF_to_DOFs,n_fdofs)
DOF_to_mDOFs = FESpaces._renumber_constraints!(DOF_to_DOFs,mDOF_to_DOF)
cellids = DofMaps._setup_cell_to_lomdof_to_omdof(
  Vact.cell_dofs_ids,DOF_to_mDOFs,DOF_to_terms,n_fdofs,n_fmdofs)


  Tp = eltype(sDOF_to_dofs.ptrs)
  Td = eltype(sDOF_to_dofs.data)
  Tc = eltype(sDOF_to_coeffs.data)
  Tt = eltype(sDOF_to_terms.data)

  DOF_to_DOFs_ptrs = ones(Tp,n_DOFs+1)

  n_sDOFs = length(sDOF_to_dof)

  for sDOF in 1:n_sDOFs
    aa = sDOF_to_dofs.ptrs[sDOF]
    bb = sDOF_to_dofs.ptrs[sDOF+1]
    dof = sDOF_to_dof[sDOF]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    DOF_to_DOFs_ptrs[DOF+1] = bb-aa
  end

  length_to_ptrs!(DOF_to_DOFs_ptrs)
  ndata = DOF_to_DOFs_ptrs[end]-1
  DOF_to_DOFs_data = zeros(Td,ndata)
  DOF_to_coeffs_data = ones(Tc,ndata)
  DOF_to_terms_data = zeros(Tt,ndata)

  for DOF in 1:n_DOFs
    q = DOF_to_DOFs_ptrs[DOF]
    DOF_to_DOFs_data[q] = DOF
    term = sDOF_to_terms.data[DOF]
    DOF_to_terms_data[q] = term
  end

  for sDOF in 1:n_sDOFs
    dof = sDOF_to_dof[sDOF]
    DOF = FESpaces._dof_to_DOF(dof,n_fdofs)
    q = DOF_to_DOFs_ptrs[DOF]-1
    pini = sDOF_to_dofs.ptrs[sDOF]
    pend = sDOF_to_dofs.ptrs[sDOF+1]-1
    for (i,p) in enumerate(pini:pend)
      _dof = sDOF_to_dofs.data[p]
      _DOF = FESpaces._dof_to_DOF(_dof,n_fdofs)
      coeff = sDOF_to_coeffs.data[p]
      term = sDOF_to_terms.data[p]
      DOF_to_DOFs_data[q+i] = _DOF
      DOF_to_coeffs_data[q+i] = coeff
      DOF_to_terms_data[q+i] = term
    end
  end

  DOF_to_DOFs = Table(DOF_to_DOFs_data,DOF_to_DOFs_ptrs)

  # ndofs = maximum(cellids.data)
  # for dof in 1:ndofs
  #   ids = findall(cellids.data.==dof)
  #   i1 = first(ids)
  #   @assert all(cellids.terms[ids].==cellids.terms[i1])
  # end
