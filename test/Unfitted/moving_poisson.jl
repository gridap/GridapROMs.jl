using DrWatson
using LinearAlgebra
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Algebra
using Gridap.FESpaces
using GridapROMs.RBSteady
using GridapROMs.Extensions
using GridapROMs.ParamAlgebra
using GridapROMs.Utils

tol_or_rank(tol,rank) = @assert false "Provide either a tolerance or a rank for the reduction step"
tol_or_rank(tol::Real,rank) = tol
tol_or_rank(tol::Real,rank::Int) = tol
tol_or_rank(tol,rank::Int) = rank

method=:pod
n=20
tol=1e-4
rank=nothing
nparams=250
nparams_res=250
nparams_jac=250

@assert method ∈ (:pod,:ttsvd) "Unrecognized reduction method! Should be one of (:pod,:ttsvd)"

pdomain = (0.5,1.5)
pspace = ParamSpace(pdomain)

R = 0.3
pmin = Point(0,0)
pmax = Point(2,2)
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
ncentroids = 16
state_reduction = LocalReduction(tolrank,energy;nparams,ncentroids)

fesolver = ExtensionSolver(LUSolver())
rbsolver = RBSolver(fesolver,state_reduction;nparams_res=nparams,nparams_jac=nparams,interp=true)

const γd = 10.0
const hd = dp[1]/n

function def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
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

  domains = FEDomains((Ω,Γ),(Ω,Γ))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

μ = realization(pspace;nparams)

feop = param_operator(μ) do μ
  println("------------------")
  def_fe_operator(μ)
end

# POD

fesnaps, = solution_snapshots(rbsolver,feop)
# red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
# jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
# red_jac = reduced_jacobian(rbsolver.jacobian_reduction,red_trial,red_test,jacs)
# ress = residual_snapshots(rbsolver,feop,fesnaps)
# red_res = reduced_residual(rbsolver.residual_reduction,red_test,ress)
# rbop = RBOperator(feop,red_trial,red_test,red_jac,red_res)
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realization(pspace;nparams=1)#10,sampling=:uniform)

x̂,rbstats = solve(rbsolver,rbop,μon)

x,festats = solution_snapshots(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,x,x̂,festats,rbstats)

# TT

_bgmodel = TProductDiscreteModel(pmin,pmax,partition)
_Ωbg = Triangulation(_bgmodel)
_dΩbg = Measure(_Ωbg,degree)

_testbg = FESpace(_Ωbg,reffe,conformity=:H1)

_energy(du,v) = ∫(v*du)_dΩbg + ∫(∇(v)⋅∇(du))_dΩbg
_tolrank = fill(tolrank,2)
_state_reduction = LocalReduction(_tolrank,_energy;nparams,ncentroids)
_rbsolver = RBSolver(fesolver,_state_reduction;nparams_res=nparams,nparams_jac=nparams,interp=true)

function _def_fe_operator(μ)
  x0 = Point(μ[1],μ[1])
  geo = !disk(R,x0=x0)
  cutgeo = cut(_bgmodel,geo)

  Ωact = Triangulation(cutgeo,ACTIVE)
  Ω = Triangulation(cutgeo,PHYSICAL)
  Γ = EmbeddedBoundary(cutgeo)

  dΩ = Measure(Ω,degree)
  dΓ = Measure(Γ,degree)

  n_Γ = get_normal_vector(Γ)

  a(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ + ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
  l(v,dΩ,dΓ) = ∫(f⋅v)dΩ + ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
  res(u,v,dΩ,dΓ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ,dΓ)

  domains = FEDomains((Ω,Γ),(Ω,Γ))

  # agfem
  strategy = AggregateAllCutCells()
  aggregates = aggregate(strategy,cutgeo)
  testact = FESpace(Ωact,reffe,conformity=:H1)
  testagg = AgFEMSpace(testact,aggregates)

  test = DirectSumFESpace(_testbg,testagg)
  trial = TrialFESpace(test,g)
  ExtensionLinearOperator(res,a,trial,test,domains)
end

_feop = param_operator(μ) do μ
  println("------------------")
  _def_fe_operator(μ)
end

_fesnaps, = solution_snapshots(_rbsolver,_feop)
_rbop = reduced_operator(_rbsolver,_feop,_fesnaps)

_x̂, = solve(_rbsolver,_rbop,μon)

_x, = solution_snapshots(_rbsolver,_feop,μon)
_perf = eval_performance(_rbsolver,_feop,_rbop,_x,_x̂,festats,rbstats)

# compare POD with TT

x0 = Point(μ.params[1][1],μ.params[1][1])
geo = !disk(R,x0=x0)
cutgeo = cut(_bgmodel,geo)

Ωact = Triangulation(cutgeo,ACTIVE)

strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)
testact = FESpace(Ωact,reffe,conformity=:H1)
testagg = AgFEMSpace(testact,aggregates)

test = DirectSumFESpace(testbg,testagg)

fdof_to_bg_fdofs,ddof_to_bg_ddofs = Extensions.get_active_dof_to_bg_dof(testbg,testagg)
bg_cell_dof_ids = Extensions.get_bg_cell_dof_ids(testagg,testbg)

_test = DirectSumFESpace(_testbg,testagg)

_fdof_to_bg_fdofs,_ddof_to_bg_ddofs = Extensions.get_active_dof_to_bg_dof(_testbg,testagg)
_bg_cell_dof_ids = Extensions.get_bg_cell_dof_ids(testagg,_testbg)

op = feop.operators[1]
_op = _feop.operators[1]

solve(fesolver.solver,op.op) ≈ solve(fesolver.solver,_op.op)

u = solve(fesolver.solver,op.op)
_u = solve(fesolver.solver,_op.op)

# u_bg = extend_solution(fesolver.extension,get_trial(op),u)
fs = get_trial(op)
fin = Extensions.get_space(fs)
uh_in = FEFunction(fin,u)
uh_in_bg = Extensions.ExtendedFEFunction(fs,u)

fout = Extensions.get_out_space(fs)
uh_out = Extensions.harmonic_extension(fout,uh_in_bg)

# Extensions.harmonic_extension(fout,uh_in_bg)
Ωout = get_triangulation(fout)
dΩout = Measure(Ωout,degree)

a(u,v) = ∫(∇(u)⊙∇(v))dΩout
l(v) = (-1)*∫(∇(uh_in_bg)⊙∇(v))dΩout
assem = SparseMatrixAssembler(fout,fout)

A = assemble_matrix(a,assem,fout,fout)
b = assemble_vector(l,assem,fout)
uout = solve(LUSolver(),A,b)

#

_fs = get_trial(_op)
_fin = Extensions.get_space(_fs)
_uh_in = FEFunction(_fin,_u)
_uh_in_bg = Extensions.ExtendedFEFunction(_fs,_u)

_fout = Extensions.get_out_space(_fs)
_uh_out = Extensions.harmonic_extension(_fout,_uh_in_bg)

_Ωout = get_triangulation(_fout)
_dΩout = Measure(_Ωout,degree)

_a(u,v) = ∫(∇(u)⊙∇(v))_dΩout
_l(v) = (-1)*∫(∇(_uh_in_bg)⊙∇(v))_dΩout
_assem = SparseMatrixAssembler(_fout,_fout)

_A = assemble_matrix(_a,_assem,_fout,_fout)
_b = assemble_vector(_l,_assem,_fout)
_uout = solve(LUSolver(),_A,_b)

# _test = DirectSumFESpace(_testbg,testagg)
space = EmbeddedFESpace(testagg,_testbg)
# complementary = complementary_space(space)
bg_space = testbg

bg_trian = get_triangulation(bg_space)
trian = get_triangulation(space)
D = num_cell_dims(trian)
glue = get_glue(trian,Val(D))
cface_to_mface = findall(x->x<0,glue.mface_to_tface)
bg_model = Extensions.get_active_model(bg_trian)
ctrian = Triangulation(bg_model,cface_to_mface)

T = Float64
order = get_polynomial_order(bg_space)
reffe = ReferenceFE(lagrangian,T,order)
_cspace = FESpace(ctrian,reffe,conformity=:H1)

bg_cell_dof_ids = get_cell_dof_ids(bg_space)
fcdof_to_bg_fcdof = Extensions.get_dofs_at_cells(bg_cell_dof_ids,cface_to_mface)
shared_dofs = intersect(fcdof_to_bg_fcdof,space.fdof_to_bg_fdofs)
fdof_to_bg_fdofs = setdiff(fcdof_to_bg_fcdof,shared_dofs)
cell_dof_ids = get_cell_dof_ids(_cspace)

isdiri = zeros(Bool,num_free_dofs(_cspace))
for ldof in eachindex(isdiri)
  bg_dof = fcdof_to_bg_fcdof[ldof]
  if bg_dof ∈ shared_dofs
    isdiri[ldof] = true
  end
end

ndiri = cumsum(isdiri)
for (idof,ldof) in enumerate(cell_dof_ids.data)
  if isdiri[ldof] == 0
    cell_dof_ids.data[idof] = ldof-ndiri[ldof]
  else
    cell_dof_ids.data[idof] = -ndiri[ldof]
  end
end

ndirichlet = isempty(ndiri) ? 0 : last(ndiri)
nfree = num_free_dofs(_cspace)-ndirichlet
