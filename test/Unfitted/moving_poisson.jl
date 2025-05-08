using DrWatson
using LinearAlgebra
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Algebra
using GridapROMs.RBSteady
using GridapROMs.Extensions

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=40
tol=1e-4
rank=nothing
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.4,0.6,0.4,0.6,1.,5.)
pspace = ParamSpace(pdomain)

R = 0.2
pmin = Point(0,0)
pmax = Point(1,1)
dp = pmax - pmin

partition = (n,n)
if method==:ttsvd
  bgmodel = TProductDiscreteModel(pmin,pmax,partition)
else
  bgmodel = CartesianDiscreteModel(pmin,pmax,partition)
end

f(x) = 1.0
g(x) = x[1]-x[2]

order = 2
degree = 2*order

Ωbg = Triangulation(bgmodel)
dΩbg = Measure(Ωbg,degree)

reffe = ReferenceFE(lagrangian,Float64,order)
testbg = FESpace(Ωbg,reffe,conformity=:H1)

energy(du,v) = ∫(v*du)dΩbg + ∫(∇(v)⋅∇(du))dΩbg
tolrank = tol_or_rank(tol,rank)
tolrank = method == :ttsvd ? fill(tolrank,2) : tolrank
state_reduction = Reduction(tolrank,energy;nparams)

X = assemble_matrix(energy,testbg,testbg)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac)

const γd = 10.0
const hd = dp[1]/n

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[2])
  geo = !disk(R,x0=x0)
  cutgeo = cut(bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ,dΓ)

  trian_a = (Ω,Γ)
  trian_res = (Ω,Γ)
  domains = FEDomains(trian_res,trian_a)

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

μ = realization(pspace;nparams=50)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

# x = solve(fesolver,feop)
# i = get_dof_map(testbg)
# fesnaps = Snapshots(x,i,μ)

# basis = reduced_basis(state_reduction,fesnaps,X)
# red_test = reduced_subspace(testbg,basis)
# red_trial = red_test

fesnaps, = solution_snapshots(rbsolver,feop)
red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)

using Gridap.FESpaces
using Gridap.Algebra
using GridapROMs.Utils

# Φ = get_basis(basis)
# fesnaps - Φ*Φ'*X*fesnaps
# μtest = realization(pspace;sampling=:uniform)
# xtest = solve(fesolver,get_fe_operator(μtest.params[1]))
# xtest - Φ*Φ'*X*xtest

ress = residual_snapshots(rbsolver,feop,fesnaps)

x0 = similar(x)
x0 = fill!(x0,0.0)
b = residual(feop,x0)
ress = Snapshots(b,i,μ)

res_red = rbsolver.residual_reduction.reduction
res_basis = reduced_basis(res_red,ress)
proj_basis = project(red_test,res_basis)
rows,interp = empirical_interpolation(res_basis)
factor = lu(interp)
domain = vector_domain(Ωbg,red_test,rows)
rhs_hypred = MDEIM(proj_basis,factor,domain)

############# ONLINE STUFF #############
μon = realization(pspace;sampling=:uniform)

x0 = Point(μon.params[1][1],μon.params[1][2])
geo = !disk(R,x0=x0)
cutgeo = cut(bgmodel,geo)

Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

n_Γ = get_normal_vector(Γ)

a(u,v) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
l(v) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
# res(u,v) = ∫(∇(v)⋅∇(u))dΩ - l(v)

feop_on = get_fe_operator(μon.params[1])
test = feop_on.op.test
############# ONLINE STUFF #############

dumfunction(Ω,rhs_hypred.domain.cells)

cells_in_Ω,icells_in_Ω = dumfunction(Ω,rhs_hypred.domain.cells)
cells_in_Γ,icells_in_Γ = dumfunction(Γ,rhs_hypred.domain.cells)
Ωv = view(Ω,cells_in_Ω)
Γv = view(Γ,cells_in_Γ)
dΩv = Measure(Ωv,degree)
dΓv = Measure(Γv,degree)
resv(u,v) = ∫(∇(v)⋅∇(u))dΩv - ( ∫(f⋅v)dΩv + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓv )

b = RBSteady.allocate_hyper_reduction(rhs_hypred,μon)
coeff = RBSteady.allocate_coefficient(rhs_hypred,μon)
bfem = RBSteady.allocate_coefficient(rhs_hypred,μon)
x̂ = zero_free_values(red_trial)

fill!(b,zero(eltype(b)))

uh = EvaluationFunction(red_trial,x̂)
v = get_fe_basis(red_test)

trian_res = (Ωv,Γv)
dc = resv(uh,v)

strian = Ωv
rhs_strian = shrink_domain(rhs_hypred,icells_in_Ω)
vecdata = collect_cell_hr_vector(red_test,dc,strian,rhs_strian)
assemble_hr_vector_add!(bfem,vecdata...)

inv_project!(bfem,rhs_strian)

# # offline
# fesnaps, = solution_snapshots(rbsolver,feop)
# rbop = reduced_operator(rbsolver,feop,fesnaps)

# # online
# μon = realization(feop;nparams=10,sampling=:uniform)
# x̂,rbstats = solve(rbsolver,rbop,μon)

# # test
# x,festats = solution_snapshots(rbsolver,feop,μon)
# perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)
# println(perf)

# opaa = get_fe_operator(μ.params[3])
# aa = solve(fesolver,opaa)
# bb = solve(fesolver,opbb)

# writevtk(Ωbg,datadir("plts/res_basis_one"),cellfields=["uh"=>FEFunction(testbg,res_basis.basis[:,1])])
# writevtk(Ωbg,datadir("plts/res_basis_two"),cellfields=["uh"=>FEFunction(testbg,res_basis.basis[:,2])])

# function categorize_cells(trian,cells)
#   all_cells_in = get_bg_cell_to_cell(trian)
#   cells_in_mask = fill(false,length(all_cells_in))
#   for cell in cells
#     if all_cells_in[cell] > 0
#       cells_in_mask[cell] = true
#     end
#   end
#   cells_in = findall(cells_in_mask)
#   cells_out = setdiff(cells,cells_in)
#   (cells_in,cells_out)
# end

function dumfunction(trian,bg_cells)
  bg_cell_to_cells = get_bg_cell_to_cell(trian)
  cells = bg_cell_to_cells[bg_cells]
  trian_icells = findall(!iszero,cells)
  trian_cells = cells[trian_icells]
  return (trian_cells,trian_icells)
end

function shrink_domain(a::MDEIM,icells)
  MDEIM(a.basis,a.interpolation,shrink_domain(a.domain,icells))
end

function shrink_domain(domain::VectorDomain,icells)
  VectorDomain(domain.cells[icells],domain.cell_irows[icells])
end
