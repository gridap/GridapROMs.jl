using Gridap
using GridapROMs

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
rbop = reduced_operator(rbsolver,feop,fesnaps)

μon = realisation(feop;nparams=10,sampling=:uniform)

x,festats = solution_snapshots(rbsolver,feop,μon)
x̂,rbstats = solve(rbsolver,rbop,μon)

using GridapROMs.ParamAlgebra
using GridapROMs.ParamDataStructures
using GridapROMs.ParamSteady
using GridapROMs.RBSteady 

using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs

using BenchmarkTools

op,r = rbop,μon
x̂ = zero_free_values(get_trial(op)(r))

nlop = parameterise(op,r)
syscache = allocate_systemcache(nlop,x̂)

# solve!(x̂,fesolver,nlop,syscache)
@btime residual!($syscache.b,$nlop,$x̂);
@btime jacobian!($syscache.A,$nlop,$x̂);
@btime solve!($x̂,$fesolver,$syscache.A,$syscache.b);

b = syscache.b
fill!(b,zero(eltype(b)))

uh = EvaluationFunction(nlop.paramcache.trial,x̂)
V = get_test(op)
v = get_fe_basis(V)

resf = get_res(op)
dc = resf(r,uh,v)
@btime $resf($r,$uh,$v);
assem = get_param_assembler(op,r)

using GridapROMs.Utils 
using Gridap.CellData

strian = get_domains(op.rhs)[1]
b_strian = b.fecache[strian];
rhs_strian = get_interpolation(op.rhs[strian]);
@btime collect_cell_hr_vector($V,$dc,$strian,$rhs_strian);
vecdata = collect_cell_hr_vector(V,dc,strian,rhs_strian);
@btime assemble_hr_array_add!($b_strian,$vecdata...);

@btime begin
  for strian in get_domains(op.rhs)
    b_strian = b.fecache[strian]
    rhs_strian = get_interpolation(op.rhs[strian])
    vecdata = collect_cell_hr_vector(V,dc,strian,rhs_strian)
    assemble_hr_array_add!(b_strian,vecdata...)
    # vecdata = collect_cell_vector_for_trian(V,dc,strian)
    # assemble_vector_add!(b.fecache[strian],assem,vecdata)
    # galerkin_projection!(b.coeff[strian],V,b.fecache[strian])
  end
end

@btime interpolate!($b,$op.rhs)

using Gridap.Arrays 
using Gridap.Fields 
function f1(cellvals,icells)
  vals_cache = array_cache(cellvals)
  for (i,celli) in enumerate(icells)
    getindex!(vals_cache,cellvals,celli)
  end
end
function f2(_cellvals,icells)
  cellvals = lazy_map(Reindex(_cellvals),icells)
  vals_cache = array_cache(cellvals)
  for cell in 1:length(cellvals)
    getindex!(vals_cache,cellvals,cell)
  end
end

# lazy_map(Reindex(vecdata[1]),vecdata[3])
k = Reindex(vecdata[1])
fi = map(testitem,(vecdata[3],))
T = return_type(k,fi...)
# lazy_map(k,T,vecdata[3])
j_to_i = vecdata[3]
i_to_maps = k.values.maps
i_to_args = k.values.args
j_to_maps = lazy_map(Reindex(i_to_maps),eltype(i_to_maps),j_to_i)
j_to_args = map(i_to_fk->lazy_map(Reindex(i_to_fk),eltype(i_to_fk),j_to_i), i_to_args)
LazyArray(T,j_to_maps,j_to_args...)

i_to_fk = i_to_args[1]
i_to_fk1 = lazy_testitem(i_to_fk)

function f3(_cellvals,icells)
  cellvals = lazy_map(FastReindex(_cellvals),icells)
  vals_cache = array_cache(cellvals)
  for cell in 1:length(cellvals)
    getindex!(vals_cache,cellvals,cell)
  end
end

@btime f1($vecdata[1],$vecdata[3]);
@btime f2($vecdata[1],$vecdata[3]);
@btime f3($vecdata[1],$vecdata[3]);