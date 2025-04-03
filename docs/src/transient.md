# Usage - Transient problem

In this example we solve a more complicated problem, namely a parameter- and time-dependent version of the Navier-Stokes equations.

## FE code 

We start by loading the necessary packages  

```julia
using Gridap
using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.NonlinearSolvers
using DrWatson
using Serialization

using GridapROMs

import Gridap.MultiField: BlockMultiFieldStyle
```

Next, we load a `DiscreteModel` from file (which can be found among the assets of the repo)

```julia
model_dir = datadir(joinpath("models","model_circle_2d.json"))
Ωₕ = DiscreteModelFromFile(model_dir)
labels = get_face_labeling(Ωₕ)
add_tag_from_tags!(labels,"dirichlet0",["walls_p","walls","cylinders_p","cylinders"])
add_tag_from_tags!(labels,"dirichlet",["inlet"])
```

Now, we introduce the space of tuples `(μ,t)` 

``` julia 
dt = 0.0025
t0 = 0.0
tf = 60*dt

pdomain = fill([1.0,10.0],3)
tdomain = t0:dt:tf
Dt = TransientParamSpace(pdomain,tdomain)
```

The main difference with respect to the steady case is that we consider as realizations sets of tuples `(μ,t)`. This allows for a much cleaner representation of the `(μ,t)`-dependence in the problem. 

!!! note

  Times are not sampled from a [`TransientParamSpace`](@ref), in the sense that we consider the sets `(μ,t) ∀ t ∈ t0:dt:tf`, where `μ` is a sampled quantity.

The way in which we simultaneously evaluate parameter- and time-dependent functions is with the structure `TransientParamFunction`, which generalizes a [`ParamFunction`](@ref) to the transient case. For example, we can consider the following Dirichlet datum for our problem

```julia
const W = 0.5
inflow(μ,t) = abs(1-cos(2π*t/tf)+μ[3]*sin(μ[2]*2π*t/tf)/100)
g_in(μ,t) = x -> VectorValue(-x[2]*(W-x[2])*inflow(μ,t),0.0)
gₚₜ_in(μ,t) = TransientParamFunction(g_in,μ,t)
g_0(x,μ,t) = x -> VectorValue(0.0,0.0)
gₚₜ_0(μ,t) = TransientParamFunction(g_0,μ,t)
```

which we use to define the FE spaces. We employ the Inf-Sup stable `P2-P1` (Taylor-Hood) pair for velocity and pressure, respectively:

```julia
order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
test_u = TestFESpace(Ωₕ,reffe_u;conformity=:H1,dirichlet_tags=["dirichlet","dirichlet0"])
trial_u = TransientTrialParamFESpace(test_u,[gμt_in,gμt_0])
reffe_p = ReferenceFE(lagrangian,Float64,order-1)
test_p = TestFESpace(Ωₕ,reffe_p;conformity=:H1)
trial_p = TransientTrialParamFESpace(test_p)
test = TransientMultiFieldParamFESpace([test_u,test_p];style=BlockMultiFieldStyle())
trial = TransientMultiFieldParamFESpace([trial_u,trial_p];style=BlockMultiFieldStyle())
```

A [`TransientTrialParamFESpace`](@ref) extends a traditional `TransientTrialFESpace` in Gridap, as it allows to provide a `(μ,t)`-dependent Dirichlet datum. The same holds for the multi-field version [`TransientMultiFieldParamFESpace`](@ref). 

!!! note

  In the multi-field scenario, the `BlockMultiFieldStyle` style should always be used. Check the appropriate documentation of [Gridap](https://github.com/gridap/Gridap.jl) for more information.

Now we introduce the information related to the numerical integration 

```julia
order = 2
degree = 2*order+1
τₕ = Triangulation(Ωₕ)
dΩₕ = Measure(τₕ,degree)
```

and then the problem's weak formulation 

```julia
const Re = 100.0
a(x,μ,t) = μ[1]/Re
a(μ,t) = x->a(x,μ,t)
aμt(μ,t) = TransientParamFunction(a,μ,t)

conv(u,∇u) = (∇u')⋅u
dconv(du,∇du,u,∇u) = conv(u,∇du)+conv(du,∇u)
c(u,v,dΩₕ) = ∫( v⊙(conv∘(u,∇(u))) )dΩₕ
dc(u,du,v,dΩₕ) = ∫( v⊙(dconv∘(du,∇(du),u,∇(u))) )dΩₕ

stiffness(μ,t,(u,p),(v,q),dΩₕ) = ∫(aμt(μ,t)*∇(v)⊙∇(u))dΩₕ - ∫(p*(∇⋅(v)))dΩₕ + ∫(q*(∇⋅(u)))dΩₕ
mass(μ,t,(uₜ,pₜ),(v,q),dΩₕ) = ∫(v⋅uₜ)dΩₕ
res(μ,t,(u,p),(v,q),dΩₕ) = ∫(v⋅∂t(u))dΩₕ + stiffness(μ,t,(u,p),(v,q),dΩₕ)

res_nlin(μ,t,(u,p),(v,q),dΩₕ) = c(u,v,dΩₕ)
jac_nlin(μ,t,(u,p),(du,dp),(v,q),dΩₕ) = dc(u,du,v,dΩₕ)
```

Note that we have split the linear terms of the Navier-Stokes equations from the nonlinear convection terms, allowing us to increase efficiency of the algorithm. Now we introduce two FE operators, one for the linear terms and the other for the nonlinear ones:

```julia
trian_res = (τₕ,)
trian_jac = (τₕ,)
trian_jac_t = (τₕ,)
domains_lin = FEDomains(trian_res,(trian_jac,trian_jac_t))
domains_nlin = FEDomains(trian_res,(trian_jac,))

feop_lin = TransientParamLinearOperator((stiffness,mass),res,ptspace,
  trial,test,domains_lin)
feop_nlin = TransientParamOperator(res_nlin,jac_nlin,ptspace,
  trial,test,domains_nlin)

feop = LinearNonlinearTransientParamOperator(feop_lin,feop_nlin)
```

Next, we define the time marching scheme for our problem, along with a suitable initial condition 

```julia 
u0(x,μ) = VectorValue(0.0,0.0)
u0(μ) = x->u0(x,μ)
u0μ(μ) = ParamFunction(u0,μ)
p0(x,μ) = 0.0
p0(μ) = x->p0(x,μ)
p0μ(μ) = ParamFunction(p0,μ)
xh0μ(μ) = interpolate_everywhere([u0μ(μ),p0μ(μ)],trial(μ,t0))

nls = NewtonSolver(LUSolver();rtol=1e-10)
θ = 1
fesolver = ThetaMethod(nls,dt,θ)
```

## GridapROMs code 

We finally discuss the code relative to the reduced part. As usual we start by defining the `RBSolver` of the problem 

```julia
coupling((du,dp),(v,q)) = ∫(dp*(∇⋅(v)))dΩₕ
energy((du,dp),(v,q)) = ∫(∇(v)⊙∇(du))dΩₕ + ∫(dp*q)dΩₕ

tol = 1e-4
state_reduction = TransientReduction(coupling,tol,energy;nparams=50,sketch=:sprn)
reduction_l = TransientMDEIMReduction(tol;nparams=40)
reduction_a = TransientMDEIMReduction(tol;nparams=20)
reduction_m = TransientMDEIMReduction(tol;nparams=1)
reduction_am = (reduction_a,reduction_m)
rbsolver = RBSolver(fesolver,state_reduction,reduction_l,reduction_am)

```

The main novelty is the use of transient reduction techniques. In particular:

* A [`TransientReduction`](@ref) provides the information related to a reduction in space, and a reduction in time. 

* A [`TransientMDEIMReduction`](@ref) provides the information related to a hyper-reduction in space, and a hyper-reduction in time. 

* Whenever we provide a `coupling` variable in the reduction strategy, a reduction of type `SupremizerReduction` is returned. This type simply acts as a wrapper for a reduction strategy (of type `TransientReduction` in our case), and has the scope of performing a supremizer enrichment for the stabilization of the reduced problem. Check [this](https://doi.org/10.1002/nme.4772) reference for more details on supremizer stabilizations. They are useful, for e.g., when reducting saddle-point problems such as the Stokes or Navier-Stokes equations.

The subsequent steps procede as in a steady problem:

```julia 
using DrWatson 
using Serialization

dir = datadir("navier-stokes")
create_dir(dir) 

try # try loading offline quantities
    rbop = load_operator(dir,feop)
catch # offline phase
    rbop = reduced_operator(rbsolver,feop)
    save(dir,rbop)
end

μon = realization(D;nparams=10,rand=true)
x̂on,rbstats = solve(rbsolver,rbop,μon)

xon,festats = solve(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,xon,x̂on,festats,rbstats)
println(perf)
```