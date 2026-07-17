# module MovingStokes

using DrWatson
using Gridap
using GridapEmbedded
using GridapROMs

using Gridap.Geometry
using Gridap.CellData
using Gridap.FESpaces
using Gridap.MultiField
using GridapROMs.RBSteady
using GridapROMs.Extensions

import Gridap.Geometry: push_normal
import GridapROMs.RBSteady: change_context
import Gridap.Helpers: @notimplementedif

method = :pod
@notimplementedif method != :pod "Must still implement TT-SVD version of this test"
tol = 1e-4
nparams = 100
compression = :local
ncentroids = 8

const L = 2
const W = 2
const n = 40
const γd = 10.0
const hd = max(L,W)/n

domain = (0,L,0,W)
partition = (n,n)
bgmodel = CartesianDiscreteModel(domain,partition)

order = 2
degree = 2*order

pdomain = (0.7,1.3,0.7,1.3)
pspace = ParamSpace(pdomain)

# quantities on the base configuration

μ0 = (1.0,1.0)
x0 = Point(μ0[1],μ0[2])
geo = !disk(0.3,x0=x0)
cutgeo = cut(bgmodel,geo)

Ωbg = Triangulation(bgmodel)
Ωact = Triangulation(cutgeo,ACTIVE)
Ω = Triangulation(cutgeo,PHYSICAL)
Γ = EmbeddedBoundary(cutgeo)

dΩbg = Measure(Ωbg,degree)
dΩ = Measure(Ω,degree)
dΓ = Measure(Γ,degree)

energy((du,dp),(v,q)) = ∫(du⋅v)dΩ + ∫(dp*q)dΩ + ∫(∇(v)⊙∇(du))dΩ + ∫((γd/hd)*du⋅v)dΓ
coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩ

n_Γ = get_normal_vector(Γ)
strategy = AggregateAllCutCells()
aggregates = aggregate(strategy,cutgeo)

domains = FEDomains((Ω,),(Ω,Γ))

g(μ) = x -> VectorValue(x[2]*(W-x[2]),0.0)*(x[1]≈0.0)
gμ(μ) = parameterise(g,μ)

reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1)

testact_u = FESpace(Ωact,reffe_u,conformity=:H1,dirichlet_tags=[1,2,3,4,5,6,7])
testact_p = FESpace(Ωact,reffe_p,conformity=:H1)
test_u = AgFEMSpace(testact_u,aggregates)
trial_u = ParamTrialFESpace(test_u,gμ)
test_p = AgFEMSpace(testact_p,aggregates)
trial_p = ParamTrialFESpace(test_p)
test = MultiFieldFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = MultiFieldFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())

function get_deformation_map(μ)
  φ(μ) = x -> VectorValue(μ[1]-x[1] + (μ[2]/μ0[2])*(x[1]-μ0[1]),μ[1]-x[2] + (μ[2]/μ0[2])*(x[2]-μ0[1]))
  φμ(μ) = parameterise(φ,μ)

  E = 1
  ν = 0.33
  λ = E*ν/((1+ν)*(1-2*ν))
  p = E/(2(1+ν))
  σ(ε) = λ*tr(ε)*one(ε) + 2*p*ε

  dΩ = Measure(Ω,2*degree)
  dΓ = Measure(Γ,2*degree)

  a(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ + ∫( (γd/hd)*v⋅u - v⋅(n_Γ⋅(σ∘ε(u))) - (n_Γ⋅(σ∘ε(v)))⋅u )dΓ
  l(μ,v) = ∫( (γd/hd)*v⋅φμ(μ) - (n_Γ⋅(σ∘ε(v)))⋅φμ(μ) )dΓ
  res(μ,u,v) = ∫( ε(v) ⊙ (σ∘ε(u)) )*dΩ - l(μ,v)

  reffeφ = ReferenceFE(lagrangian,VectorValue{2,Float64},2*order)
  Vφact = FESpace(Ωact,reffeφ,conformity=:H1,dirichlet_tags="boundary")
  Vφ = AgFEMSpace(Vφact,aggregates)
  Uφ = ParamTrialFESpace(Vφ)

  feop = LinearParamOperator(res,a,pspace,Uφ,Vφ)
  d, = solve(LUSolver(),feop,μ)
  FEFunction(Uφ(μ),d)
end

function def_fe_operator(μ)
  φh = get_deformation_map(μ)
  Ωactφ = mapped_grid(Ωact,φh)

  ϕ = get_cell_map(Ωactφ)
  ϕh = GenericCellField(ϕ,Ωact,ReferenceDomain())
  dJ = det∘(∇(ϕh))
  dJΓn(j,c,n) = j*√(n⋅inv(c)⋅n)
  C = (j->j⋅j')∘(∇(ϕh))
  invJt = inv∘∇(ϕh)
  ∇_I(a) = dot∘(invJt,∇(a))
  tr∇_I(a) = tr(∇_I(a))
  _n_Γ = push_normal∘(invJt,n_Γ)
  dJΓ = dJΓn∘(dJ,C,_n_Γ)
  ∫_Ω(a) = ∫(a*dJ)
  ∫_Γ(a) = ∫(a*dJΓ)

  a(μ,(u,p),(v,q),dΩ,dΓ) =
    ∫_Ω( ∇_I(v)⊙∇_I(u) - q*(tr∇_I(u)) - (tr∇_I(v))*p ) * dΩ +
    ∫_Γ( (10*γd/hd)*v⋅u - v⋅(_n_Γ⋅∇_I(u)) - (_n_Γ⋅∇_I(v))⋅u + (p*_n_Γ)⋅v + (q*_n_Γ)⋅u ) * dΓ

  res(μ,(u,p),(v,q),dΩ) = ∫_Ω( ∇_I(v)⊙∇_I(u) - q*(tr∇_I(u)) - (tr∇_I(v))*p ) * dΩ

  LinearParamOperator(res,a,pspace,trial,test,domains)
end

function local_solver(rbsolver,rbop,μ,x,festats)
  gsolver = change_context(rbsolver)
  k, = get_clusters(rbop.test)
  μsplit = cluster(μ,k)
  xsplit = cluster(x,k)
  perfs = ROMPerformance[]
  for (μi,xi) in zip(μsplit,xsplit)
    feopi = def_fe_operator(μi)
    rbopi = change_operator(get_local(rbop,first(μi)),feopi)
    x̂,rbstats = solve(gsolver,rbopi,μi)
    perf = eval_performance(rbsolver,rbopi,xi,x̂,festats,rbstats)
    push!(perfs,perf)
  end
  return RBSteady.mean(perfs)
end

fesolver = LUSolver()
hr = compression == :global ? HyperReduction : LocalHyperReduction
state_reduction = SupremizerReduction(coupling,tol,energy;nparams,sketch=:sprn,compression,ncentroids)
residual_reduction = hr(tol.*1e-2;nparams,ncentroids)
jacobian_reduction = hr(tol.*1e-2;nparams,ncentroids)
rbsolver = RBSolver(fesolver,state_reduction,residual_reduction,jacobian_reduction)

μ = realisation(pspace;nparams)
feop = def_fe_operator(μ)
fesnaps, = solution_snapshots(rbsolver,feop,μ)

μon = realisation(pspace;nparams=10,sampling=:uniform)
feopon = def_fe_operator(μon)
x,festats = solution_snapshots(rbsolver,feopon,μon)

rbop = reduced_operator(rbsolver,feop,fesnaps)

if compression == :global
  rbop′ = change_operator(rbop,feopon)
  x̂,rbstats = solve(rbsolver,rbop′,μon)
  perf = eval_performance(rbsolver,rbop′,x,x̂,festats,rbstats)
else
  perf = local_solver(rbsolver,rbop,μon,x,festats)
end

println(perf)

# ─── minimal reproduction of the Int32/Int64 error ────────────────────────────
# Replicates x̂,rbstats = solve(gsolver,rbopi,μi) at moving_stokes.jl:143,
# tracing only the leaf calls in GridapROMs before the Gridap testitem failure.

gsolver_rep = change_context(rbsolver)
k_rep, = get_clusters(rbop.test)
μi_rep = first(cluster(μon,k_rep))
feopi_rep = def_fe_operator(μi_rep)
rbopi_rep = change_operator(get_local(rbop,first(μi_rep)),feopi_rep)

# solve(gsolver,rbopi,μi) → parameterise + allocate
nlop_rep = parameterise(rbopi_rep,μi_rep)
x̂_rep = zero_free_values(get_trial(rbopi_rep)(μi_rep))

# solve!(x̂,fesolver,nlop,syscache) → residual!(b,nlop,x̂)
#   → residual!(b,rbopi,μi,x̂,paramcache)  [ReducedOperators.jl:182]
paramcache_rep = nlop_rep.paramcache
uh_rep = EvaluationFunction(paramcache_rep.trial,x̂_rep)
test_rep = get_test(rbopi_rep)
v_rep = get_fe_basis(test_rep)
dc_rep = get_res(rbopi_rep)(μi_rep,uh_rep,v_rep)

# collect_cell_hr_vector [HRAssemblers.jl:19-33]
strian_rep = first(get_domains_res(rbopi_rep))
rhs_strian_rep = get_interpolation(RBSteady.get_rhs(rbopi_rep)[strian_rep])
scell_vec_rep = get_contribution(dc_rep,strian_rep)
cell_vec_rep,trian_rep2 = move_contributions(scell_vec_rep,strian_rep)
# move_contributions(arr,strian) is a no-op (returns (arr,strian) unchanged)
# so trian_rep2 === strian_rep

# attach_constraints_rows(test,cell_vec,trian) [FESpaceInterface.jl:361]
# ConstraintStyle(test_rep) == Constrained() because MultiFieldRBSpace wraps AgFEMSpaces
# → attach_constraints_rows(test,cell_vec,trian,::Constrained) [FESpaceInterface.jl:369]
# → get_cell_constraints(test,trian)              [FESpaceInterface.jl:370]
# → get_cell_fe_data(get_cell_constraints,test,trian) [FESpaceInterface.jl:325-326]

# unroll get_cell_fe_data(fun,f,ttrian) [FESpaceInterface.jl:20-30]
trian_rep2 = strian_rep
sface_to_data_rep = get_cell_constraints(test_rep)   # unary: constraints on test's own trian
strian_inner_rep  = get_triangulation(test_rep)       # the source triangulation (e.g. Ωact)
D_rep      = num_cell_dims(strian_inner_rep)
sglue_rep  = get_glue(strian_inner_rep,Val(D_rep))    # FaceToFaceGlue for source space
tglue_rep  = get_glue(trian_rep2,Val(D_rep))          # FaceToFaceGlue for target (HR) trian
# trian_rep2 is a ChildTriangulation whose get_glue calls
#   view(parent_glue, child.cell_to_parent_cell)
# which builds tface_to_mface = lazy_map(Reindex(parent.tface_to_mface), Int64_ids)
# eltype declared as Int32 (from parent), but Int64_ids come from findall(::Vector{Bool})

# unroll get_cell_fe_data(fun,sface_to_data,sglue,tglue) [FESpaceInterface.jl:33-38]
mface_to_sface_rep  = sglue_rep.mface_to_tface
tface_to_mface_rep  = tglue_rep.tface_to_mface   # LazyArray{…,Int32} whose elements are Int64
mface_to_data_rep   = extend(sface_to_data_rep,mface_to_sface_rep)
# ↓ this is FESpaceInterface.jl:37 — crashes because
#   lazy_map calls testitem(tface_to_mface_rep)
#   testitem does first(tface_to_mface_rep)::Int32 but first() returns Int64
tface_to_data_rep   = lazy_map(Reindex(mface_to_data_rep),tface_to_mface_rep)

tglue_rep  = get_glue(trian_rep2.parent,Val(D_rep)) 
testitem(tglue_rep.tface_to_mface)

a = AppendedArray([1,2],Int32[3,4])
av = lazy_map(Reindex(a),[3,4])
testitem(av)

cache = array_cache(av)
getindex!(cache,av,2)

b = AppendedArray(Int32[1,2],Int32[3,4])
bv = lazy_map(Reindex(b),[3,4])
bcache = array_cache(bv)

fi = map(testitem,([3,4],))
T = return_type(Reindex(a), fi...)
lazy_map(Reindex(a),T,[3,4])

# function Arrays.lazy_map(
#   k::Reindex{<:AppendedArray},
#   j_to_i::AbstractArray
#   )

#   j_to_ia,j_to_ib = _split_ids(j_to_i,k.values)
#   if length(j_to_ia) == 0
#     return lazy_map(Reindex(k.values.b),j_to_ib)
#   elseif length(j_to_ib) == 0
#     return lazy_map(Reindex(k.values.a),j_to_ia)
#   else
#     aj = lazy_map(Reindex(k.values.a),j_to_ia)
#     bj = lazy_map(Reindex(k.values.b),j_to_ib)
#     return lazy_append(aj,bj)
#   end
# end

# function _split_ids(j_to_i,aa)
#   na = 0 
#   for i in j_to_i 
#     if i <= length(aa.a)
#       na += 1
#     else
#       break
#     end
#   end
#   j_to_ia = zeros(eltype(j_to_i),na)
#   j_to_ib = zeros(eltype(j_to_i),length(j_to_i)-na)
#   for (j,i) in enumerate(j_to_i)
#     if i <= length(aa.a)
#       j_to_ia[j] = i
#     else
#       j_to_ib[j-na] = i - length(aa.a)  
#     end
#   end
#   return j_to_ia,j_to_ib
# end