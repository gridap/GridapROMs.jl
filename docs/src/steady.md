# Usage - Steady problem

## Installation

ROM is a registered package. You can install it by running:

```julia
# Use ] to enter the Pkg REPL mode
pkg> add ROM
```

## Load package

Load the package normally with

```julia
using ROM
```

## Workflow 

The workflow of the package is analogous to that of [Gridap](https://github.com/gridap/Gridap.jl), a library for the grid-based approximation of PDEs. Comprehensive Gridap tutorials can be found [here](https://gridap.github.io/Tutorials/stable). We load Gridap with 

```julia
using Gridap
```

## Manufactured solution 

In this example we solve a parameter dependent Poisson equation 

```julia
-ν*Δu  = f in Ω
ν*∇u⋅n = h in Γn
```

where `Ω` is a sufficiently regular spatial domain, `ν` is a (positive) conductivity coefficient, `u` is the problem's unknown, `f` is a forcing term, and `h` a Neumann datum defined on the Neumann boundary `Γn`. In this problem, we consider `Ω = [0,1]^2` and `Γn` to be the right leg of the square. The remaining boundary is Dirichlet, and here we impose a manufactured, parameter-dependent solution. We consider the problem given by the following data: 

```julia
ν(μ) = exp(-sum(μ))
u(μ) = x -> μ[1]*x[1] + μ[2]*x[2]
f(μ) = x -> -ν(μ)*Δ(u(x,μ))
h(μ) = x -> 1
```

Next, we parameterize the data defined above exclusively by `μ` in the following manner:

```julia
uₚ(μ) = ParamFunction(u,μ)
fₚ(μ) = ParamFunction(f,μ)
hₚ(μ) = ParamFunction(h,μ)
```

A [`ParamFunction`](@ref) is a function that can be evaluated efficiently for any number of desired parameters. In a steady setting, it takes as argument a function (`u`, `f` and `h` in the cases above) and a parameter variable. In a transient setting, an additional time variable must be included (more details can be found in the following tutorial for transient problems).

## Geometry 

We define the geometry of the problem using 

```julia
Ω = (0,1,0,1)
partition = (10,10)
Ωₕ = CartesianDiscreteModel(Ω,partition)
```

## FE spaces 

Once the discrete geometry is introduced, we define a tuple of trial, test spaces `(U,V)` as

```julia
order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
V = TestFESpace(Ωₕ,reffe;dirichlet_tags=[1,3,5,6,7])
U = TrialParamFESpace(V,uₚ)
```

A `TrialParamFESpace` extends a traditional `TrialFESpace` in Gridap, as it allows to provide a `μ`-dependent Dirichlet datum. The tags provided occupy the left, upper and bottom legs of the square (extrema excluded for the upper and bottom legs).

## Space of parameters 

We define a space of parameters, in this case `[1,10]^2`:

```julia 
D = ParamSpace((1,10,1,10))
```

A parameter, in our case a 2-dimensional vector, is a single realization from `D`. By default, a parameter is sampled according to a [Halton](https://en.wikipedia.org/wiki/Halton_sequence) sequence. Other sampling methods can be defined by providing appropriate keyword arguments: 

```julia
ParamSpace((1,10,1,10),sampling=:latin_hypercube)
ParamSpace((1,10,1,10),sampling=:normal)
ParamSpace((1,10,1,10),sampling=:uniform)
ParamSpace((1,10,1,10),sampling=:uniform_tensorial)
```

A single parameter is sampled from `D` by calling

```julia
μ = realization(D) 
```

whereas a collection of 10 realizations can be obtained by running

```julia
realization(D;nparams=10)
``` 

## Numerical integration 

Before introducing the weak formulation of the problem, we define the quantities needed for the numerical integration 

```julia 
degree = 2*order
τₕ = Triangulation(Ωₕ)
Γₕ = BoundaryTriangulation(Ω;tags=[2,4,8])
dΩₕ = Measure(τₕ,degree)
dΓₕ = Measure(Γₕ,degree)
```

The physical entities corresponding to the tags provided when defining `Γₕ` are: the bottom right vertex (2), the top right vertex (4) and the interior of the right edge (8).

## Weak formulation 

Multiplying the Poisson equation by a test function `v ∈ V` and integrating by parts yields the weak formulation of the problem, whose left- and right-hand (LHS & RHS) side are

```julia
a(μ,u,v,dΩₕ) = ∫(ν(μ)*∇(v)⋅∇(u))dΩₕ 
l(μ,v,dΩₕ,dΓₕ) = a(μ,u,v,dΩₕ) - ∫(fₚ(μ)*v)dΩₕ - ∫(hₚ(μ)*v)dΓₕ
```

Note that, in contrast to a traditional Gridap code, the measures involved in the forms are passed as arguments to the forms themselves. (This prevents us from defining a FE operator just by defining a bilinear form for the LHS and a linear form for the RHS as in Gridap: we must actually write the full expression of the residual). 

## Parametric FE problem 

At this point, we can build a FE operator representing the Poisson equation: 

```julia 
τₕ_l = (Ωₕ,Γₕ)
τₕ_a = (Ωₕ,)
domains = FEDomains(τₕ_l,τₕ_a)
feop = ParamLinearFEOperator(l,a,D,U,V,domains)
```

The structure [`FEDomains`](@ref) collects the triangulations relative to the LHS & RHS. With respect to a traditional FE operator in Gridap, a [`ParamLinearFEOperator`](@ref) provides the aforementioned `FEDomains` for the LHS & RHS, as well as the parametric domain `D`.

## FE solver 

We define the FE solver for our Poisson problem: 

``` julia
ls = LUSolver()
solver = LinearFESolver(ls)
```

## RB solver

Finally, we are ready to begin the ROM part. The first part consists in defining the problem's [`RBSolver`](@ref), i.e. the reduced counterpart of a FE solver:

```julia 
tol = 1e-4
inner_prod(u,v) = ∫(∇(v)⋅∇(u))dΩₕ

reduction_sol = PODReduction(tol,inner_prod;nparams=20)
reduction_l = MDEIMReduction(tol;nparams=10)
reduction_a = MDEIMReduction(tol;nparams=10)
rbsolver = RBSolver(solver,reduction_sol,reduction_l,reduction_a)
```

A `RBSolver` contains the following information: 

* The FE solver `solver`

* The reduction strategy for the solution `reduction_sol`. This information is used to build a projection map representing a low-dimensional approximation subspace (i.e. a trial space) for our differential problem. In the case above, we use a truncated POD with tolerance `tol` on a set of 20 snapshots. The output is orthogonal with respect to the form `inner_prod`, i.e. the `H^1_0` product. 

* The hyper-reduction strategy for the residual `reduction_l`. This information is used to build a projection map representing a low-dimensional subspace for the residual, equipped with a reduced integration domain obtained via MDEIM, i.e. the matrix-based empirical interpolation method. A total of 10 residual snapshots is used to compute the output.

* Similarly, the hyper-reduction strategy for the Jacobian `reduction_a`.

!!! note 
    Other reduction strategies include fixing the rank of the truncation

    ```julia 
        rank = 5
        PODReduction(rank,inner_prod;nparams=20)
    ```

    and using randomized POD algorithms

    ```julia 
        PODReduction(tol,inner_prod;nparams=20,sketch=:sprn)
    ```

    A comprehensive documentation on randomized POD algorithms can be found [here](https://github.com/JuliaLinearAlgebra/LowRankApprox.jl).

## Offline phase 

The offline phase is the part of the code where the projection maps for the solution, LHS & RHS are computed. This phase is quite expensive to run, but on the upside it can just be run once, provided the outputs are saved to file. For this purpose, we load the well-known packages [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/) and [Serialization](https://docs.julialang.org/en/v1/stdlib/Serialization/)

```julia 
using DrWatson 
using Serialization
```

and define a saving directory for our example

```julia 
dir = datadir("poisson")
create_dir(dir) 
```

Next, we try loading the offline quantities; if the load fails, we must run the offline phase 

```julia 
try # try loading offline quantities
    rbop = load_operator(dir,feop)
catch # offline phase
    rbop = reduced_operator(rbsolver,feop)
    save(dir,rbop)
end
```

The load might fail, for e.g., if 

* It is the first time running the code

* A different directory was used to save the offline structures in previous runs 

* For developers, if the definition of one of the loaded types has changed since it was saved to file 

The offline structures are completely contained in the variable `rbop`, which is the reduced version of the FE operator `feop`. To understand better the meaning of this variable, we report the content of the function `reduced_operator`:

```julia
# compute the solution snapshots 
fesnaps, = solution_snapshots(rbsolver,feop) 
# compute the reduced trial and test spaces 
Û,V̂ = reduced_spaces(rbsolver,feop,fesnaps)
# compute the hyper-reduction for LHS & RHS
op = get_algebraic_operator(feop)
â,l̂ = reduced_weak_form(rbsolver,op,Û,V̂,fesnaps)

# fetch the reduced FEDomains
τₕ_l̂,τₕ_â = get_domains(l̂),get_domains(â)
# replace the original FEDomains with the reduced ones  
op′ = change_domains(op,τₕ_l̂,τₕ_â)
# definition of reduced operator 
rbop = GenericRBOperator(op′,Û,V̂,â,l̂)
```

## Online phase 

This step consists in computing the ROM approximation for any desired parameter. We consider, for e.g., 10 parameters distributed uniformly on `D`

```julia
μon = realization(D;nparams=10,rand=true)
```

and we solve the reduced problem 

```julia
x̂on,rbstats = solve(rbsolver,rbop,μon)
```

## Post processing 

In order to test the quality of the approximation `x̂on`, we can run the following post-processing code 

```julia
xon,festats = solve(rbsolver,feop,μon)
perf = eval_performance(rbsolver,feop,rbop,xon,x̂on,festats,rbstats,μon)
println(perf)
```

In other words, we first compute the HF solution `xon` in the online parameters `μon`, and then we run the performance tester, which in particular returns

* The relative error ||xon - x̂on|| / ||xon||, averaged on the 10 parameters in `μon`. The norm is the one specified by `inner_prod`, so in our case the `H^1_0` product.

* The speedup in terms of time and memory achieved with respect to the HF simulations. This is done by comparing the variables `rbstats` and `festats`, which contain the time (in seconds) and memory allocations (in Gb) of the two algorithms.