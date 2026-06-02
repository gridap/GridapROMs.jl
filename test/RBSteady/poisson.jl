module PoissonEquation

using Gridap
using GridapROMs

function main(
  method=:pod,compression=:global,hypred_strategy=:mdeim;
  tol=1e-4,nparams=50,nparams_res=floor(Int,nparams/3),
  nparams_jac=floor(Int,nparams/4),sketch=:sprn,ncentroids=2
  )

  method = method ∈ (:pod,:ttsvd) ? method : :pod
  compression = compression ∈ (:global,:local) ? compression : :global
  hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

  println("Running test with $compression ($method, $hypred_strategy) strategy")

  pdomain = (1,10,1,10,1,10)
  pspace = ParamSpace(pdomain)

  domain = (0,1,0,1)
  partition = (10,10)
  if method==:ttsvd
    model = TProductDiscreteModel(domain,partition)
  else
    model = CartesianDiscreteModel(domain,partition)
  end

  order = 2
  degree = 2*order

  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)
  Γn = BoundaryTriangulation(model,tags=[8])
  dΓn = Measure(Γn,degree)

  a(μ) = x -> exp(-x[1]/sum(μ))
  aμ(μ) = parameterise(a,μ)

  f(μ) = x -> 1.
  fμ(μ) = parameterise(f,μ)

  g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
  gμ(μ) = parameterise(g,μ)

  h(μ) = x -> abs(cos(μ[3]*x[2]))
  hμ(μ) = parameterise(h,μ)

  stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
  rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
  res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

  trian_res = (Ω,Γn)
  trian_stiffness = (Ω,)
  domains = FEDomains(trian_res,trian_stiffness)

  energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

  reffe = ReferenceFE(lagrangian,Float64,order)
  test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
  trial = ParamTrialFESpace(test,gμ)

  if method == :pod
    state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
  elseif method == :ttsvd
    state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
  end

  fesolver = LUSolver()
  rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

  feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
  fesnaps, = solution_snapshots(rbsolver,feop)
  rbop = reduced_operator(rbsolver,feop,fesnaps)

  μon = realisation(feop;nparams=10,sampling=:uniform)
  x̂,rbstats = solve(rbsolver,rbop,μon)
  x,festats = solution_snapshots(rbsolver,feop,μon)
  perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)

  println(perf)
end

for method in (:pod,:ttsvd), compression in (:local,:global), hypred_strategy in (:mdeim,:sopt,:rbf,:none,:affine)
  main(method,compression,hypred_strategy)
end

end


using Gridap
using GridapROMs
using GridapROMs.RBSteady

method=:pod
compression=:global
hypred_strategy=:mdeim
tol=1e-4
nparams=50
nparams_res=floor(Int,nparams/3)
nparams_jac=floor(Int,nparams/4)
sketch=:sprn
ncentroids=2

method = method ∈ (:pod,:ttsvd) ? method : :pod
compression = compression ∈ (:global,:local) ? compression : :global
hypred_strategy = hypred_strategy ∈ (:mdeim,:sopt,:rbf,:none,:affine) ? hypred_strategy : :mdeim

println("Running test with $compression ($method, $hypred_strategy) strategy")

pdomain = (1,10,1,10,1,10)
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (10,10)
if method==:ttsvd
  model = TProductDiscreteModel(domain,partition)
else
  model = CartesianDiscreteModel(domain,partition)
end

order = 2
degree = 2*order

Ω = Triangulation(model)
dΩ = Measure(Ω,degree)
Γn = BoundaryTriangulation(model,tags=[8])
dΓn = Measure(Γn,degree)

a(μ) = x -> exp(-x[1]/sum(μ))
aμ(μ) = parameterise(a,μ)

f(μ) = x -> 1.
fμ(μ) = parameterise(f,μ)

g(μ) = x -> μ[1]*exp(-x[1]/μ[2])
gμ(μ) = parameterise(g,μ)

h(μ) = x -> abs(cos(μ[3]*x[2]))
hμ(μ) = parameterise(h,μ)

stiffness(μ,u,v,dΩ) = ∫(aμ(μ)*∇(v)⋅∇(u))dΩ
rhs(μ,v,dΩ,dΓn) = ∫(fμ(μ)*v)dΩ + ∫(hμ(μ)*v)dΓn
res(μ,u,v,dΩ,dΓn) = stiffness(μ,u,v,dΩ) - rhs(μ,v,dΩ,dΓn)

trian_res = (Ω,Γn)
trian_stiffness = (Ω,)
domains = FEDomains(trian_res,trian_stiffness)

energy(du,v) = ∫(v*du)dΩ + ∫(∇(v)⋅∇(du))dΩ

reffe = ReferenceFE(lagrangian,Float64,order)
test = TestFESpace(Ω,reffe;conformity=:H1,dirichlet_tags=[1,3,7])
trial = ParamTrialFESpace(test,gμ)

if method == :pod
  state_reduction = Reduction(tol,energy;nparams,sketch,compression,ncentroids)
elseif method == :ttsvd
  state_reduction = Reduction(fill(tol,3),energy;nparams,sketch,compression,ncentroids)
end

fesolver = LUSolver()
rbsolver = RBSolver(fesolver,state_reduction;nparams_res,nparams_jac,hypred_strategy)

feop = LinearParamOperator(res,stiffness,pspace,trial,test,domains)
fesnaps, = solution_snapshots(rbsolver,feop)

red_trial,red_test = reduced_spaces(rbsolver,feop,fesnaps)
du,v = get_trial_fe_basis(red_trial),get_fe_basis(red_test)
μ = realisation(feop;nparams=10) 
# ∫(aμ(μ)*∇(v)⋅∇(du))dΩ
# integrate(aμ(μ)*∇(v)⋅∇(du),dΩ.quad)
x = get_cell_points(dΩ.quad)
cf = aμ(μ)*∇(v)⋅∇(du)
acf = cf.args[1].args[1]

reffe = ReferenceFE(lagrangian,Float64,order)
spacel2 = TestFESpace(Ω,reffe,conformity=:L2)
spaceh1 = TestFESpace(Ω,reffe,conformity=:H1)

fh = interpolate(x->x[1]+x[2],spacel2)

using Gridap.FESpaces
conformity = :L2
cell_reffe = ReferenceFE(model,reffe)
reffe_name,reffe_args,reffe_kwargs = reffe
trian = Ω
# ReferenceFE(trian,reffe_name,reffe_args...;reffe_kwargs...)
using Gridap.Geometry
using Gridap.ReferenceFEs
ctype_to_polytope = get_polytopes(trian)
cell_to_ctype = get_cell_type(trian)
# ctype_to_reffe = map(p->ReferenceFE(p,reffe_name,reffe_args...;reffe_kwargs...),ctype_to_polytope)
# ReferenceFE(ctype_to_polytope[1],reffe_name,reffe_args...;reffe_kwargs...)
cell_to_reffe = expand_cell_data(ctype_to_reffe,cell_to_ctype)

# conf = FESpaces.Conformity(testitem(cell_reffe),conformity)
# cell_fe = CellFE(model,cell_reffe,conf)

# FESpaces.get_cell_shapefuns_and_dof_basis(model,cell_reffe,conf)

gf = dc[Ω].args[1].args[1].args[1].maps[1].data[1]

fields = dc[Ω].args[1].args[1].args[1].maps
qspace = QuadratureFESpace(trian,reffe)
qdofs = get_fe_dof_basis(qspace)
cell_fields = GenericCellField(fields,trian,DomainStyle(qdofs))
cell_values = qdofs(cell_fields)

plength = param_length(first(fields))
free_values = parameterise(zero_free_values(qspace),plength)
diri_values = parameterise(zero_dirichlet_values(qspace),plength)
gather_free_and_dirichlet_values!(free_values,diri_values,qspace,cell_values)

red = Reduction(1e-5;sketch=:sprn)
_reduced_free_values = reduction(red,get_all_data(free_values))
reduced_free_values = ConsecutiveParamArray(_reduced_free_values)
rplength = param_length(reduced_free_values)
reduced_diri_values = parameterise(zero_dirichlet_values(qspace),rplength)
rcv = scatter_free_and_dirichlet_values(qspace,reduced_free_values,reduced_diri_values)
rrcv = Fill(rcv,fields.axes)
rcf = GenericCellField(rrcv,trian,PhysicalDomain())
# reduce_integral(dc[Ω],trian,order;red)

rdc = ∫(rcf*∇(v)⋅∇(du))dΩ
rdc = ∫(rrcv*∇(v)⋅∇(du))dΩ

struct ConstantCellField{DS} <: CellField
  cell_field::AbstractArray
  trian::Triangulation
  domain_style::DS
  function ConstantCellField(
    cell_field::AbstractArray,
    trian::Triangulation,
    domain_style::DomainStyle)

    DS = typeof(domain_style)
    new{DS}(Fields.MemoArray(cell_field),trian,domain_style)
  end
end

CellData.get_data(f::ConstantCellField) = f.cell_field
CellData.get_triangulation(f::ConstantCellField) = f.trian
CellData.DomainStyle(::Type{ConstantCellField{DS}}) where DS = DS()

function Arrays.evaluate!(cache,f::ConstantCellField,x::CellPoint)
  @check get_triangulation(f) == get_triangulation(x) 
  get_data(f)
end