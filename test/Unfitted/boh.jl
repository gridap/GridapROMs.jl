v = fesnaps[:,end]
uh = FEFunction(testbg,v)
writevtk(Ωbg,datadir("plts/sol"),cellfields=["uh"=>uh])

jacs = jacobian_snapshots(rbsolver,feop,fesnaps)
ress = residual_snapshots(rbsolver,feop,fesnaps)

μon = realization(pspace;nparams=1,sampling=:uniform)
feopon = param_operator(μon) do μ
  println("------------------")
  def_fe_operator(μ)
end
xon, = solution_snapshots(rbsolver,feopon,μon)

k = RBSteady.get_clusters(red_trial)
red_trialμ = RBSteady.get_local(red_trial,μon)[1]
Φμ = get_basis(red_trialμ)

xon - Φμ*Φμ'*X*xon

μtest = Realization([[0.58,]])
feoptest = UnCommonParamOperator([def_fe_operator(μtest.params[1])],μtest)
stest, = solution_snapshots(rbsolver,feoptest,μtest)
rtest = residual_snapshots(rbsolver1,feoptest,stest)
jtest = jacobian_snapshots(rbsolver1,feoptest,stest)
k = RBSteady.get_clusters(red_test)
lab = RBSteady.get_label(k,μtest.params[1])
resvec = RBSteady.cluster_snapshots(ress,k)
jacvec = RBSteady.cluster_snapshots(jacs,k)

s,V = resvec[lab],RBSteady.local_values(red_test)[lab]
hr = HRProjection(rbsolver.residual_reduction.reduction,s,V)
proj = projection(rbsolver.residual_reduction.reduction.reduction,s)

coeff = interpolate(hr.interpolation,μtest)
r̂test = get_basis(proj)*coeff[1]

maximum(abs.(rtest - r̂test))

s,U = jacvec[lab],RBSteady.local_values(red_trial)[lab]
hr = HRProjection(rbsolver.jacobian_reduction.reduction,s,U,V)
proj = projection(rbsolver.jacobian_reduction.reduction.reduction,s)
Φ = proj.basis.data

coeff = interpolate(hr.interpolation,μtest)
ĵtest = sum(map(i -> get_basis(proj)[i,i]*coeff[1][i],eachindex(coeff[1])))
Jtest = recast(jtest,jtest.dof_map.sparsity)[1]
maximum(abs.(Jtest - ĵtest))

using Plots
heatmap(Jtest - ĵtest)
E = Jtest - ĵtest

sparsity = get_common_sparsity(copy(Jtest),copy(jacs.dof_map.sparsity.matrix))
fill!(sparsity.nzval,1.0)
J = Jtest + sparsity
J.nzval .-= 1.0

J.nzval - Φ*Φ'*J.nzval

inds,interp = empirical_interpolation(proj)
factor = lu(interp)
r = get_realization(s)
red_data = RBSteady.get_at_domain(s,inds)
coeff = allocate_coefficient(proj,r)
ldiv!(coeff,factor,red_data)
Interpolator(r,coeff,strategy)

rows,cols = inds
sparsity = param_getindex(proj.basis,1)
inds = DofMaps.sparsify_split_indices(rows,cols,sparsity)
data = get_all_data(proj.basis)
datav = view(data,inds,:)
ConsecutiveParamArray(datav)

CIAO
# rh = FEFunction(testbg,rtest[:,1])
# r̂h = FEFunction(testbg,r̂test[:,1])
# eh = FEFunction(testbg,abs.(ŝtest[:,1]-stest[:,1]))
# writevtk(Ωbg,datadir("plts/res"),cellfields=["rh"=>rh,"r̂h"=>r̂h,"eh"=>eh])

# xres = fill!(copy(get_param_data(solution_snapshots(rbsolver,feopon,μtest)[1])),0.)
# # residual(feop,μtest,xres)

# x0 = Point(1.0,1.0)
# geo = !disk(R,x0=x0)
# cutgeo = cut(bgmodel,geo)

# Ωact = Triangulation(cutgeo,ACTIVE)
# Ω = Triangulation(cutgeo,PHYSICAL)
# Γ = EmbeddedBoundary(cutgeo)

# dΩ = Measure(Ω,degree)
# dΓ = Measure(Γ,degree)

# n_Γ = get_normal_vector(Γ)

# a(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ #+ ∫( (γd/hd)*v*u  - v*(n_Γ⋅∇(u)) - (n_Γ⋅∇(v))*u )dΓ
# l(v,dΩ) = ∫(f⋅v)dΩ #+ ∫( (γd/hd)*v*g - (n_Γ⋅∇(v))*g )dΓ
# res(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ - l(v,dΩ)

# trian_a = (Ω,)
# trian_res = (Ω,)
# domains = FEDomains(trian_res,trian_a)

# # agfem
# strategy = AggregateAllCutCells()
# aggregates = aggregate(strategy,cutgeo)
# testact = FESpace(Ωact,reffe,conformity=:H1,dirichlet_tags=[1,3,7])
# testagg = AgFEMSpace(testact,aggregates)
# trialagg = TrialFESpace(testagg,g)

# AA = assemble_matrix((u,v)->a(u,v,dΩ),trialagg,testagg)
