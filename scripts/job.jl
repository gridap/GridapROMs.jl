using DrWatson
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using Gridap.ODEs
using GridapROMs
using GridapROMs.Utils 
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures

include("../examples/ExamplesInterface.jl")

nx = 20
ny = 5
order = 2

# Fixed parameters
Lb = 12.5
mᵨ = 8.36
EI₁ = 47100.0
EI₂ = 471.0
β = 0.2
H = 1.1
α = 0.249
ξ = 0.0

# Domain size
Ld = Lb # damping zone length
LΩ = 2Ld + 2Lb
x₀ = 0.0
xdᵢₙ = x₀ + Ld
xb₀ = xdᵢₙ + Lb/2
xbⱼ = xb₀ + β*Lb
xb₁ = xb₀ + Lb
xdₒᵤₜ = LΩ - Ld

# Physics
g = 9.81
ρ = 1025.0
d₀ = mᵨ/ρ
a₁ = EI₁/ρ
a₂ = EI₂/ρ
kᵣ = ξ*a₁/Lb

# Numerics constants
nx_total = Int(ceil(nx/β)*ceil(LΩ/Lb))
h = LΩ / nx_total
γ = 1.0*order*(order-1)/h
βₕ = 0.5
η₀ = 0.01

# Time discretization (fixed Δt based on reference wavelength α*Lb)
_λ_ref = α*Lb
_k_ref = 2π/_λ_ref
_ω_ref = √(g*_k_ref*tanh(_k_ref*H))
_T_ref = 2π/_ω_ref
γₜ = 0.5
βₜ = 0.25
t₀ = 0.0
dt = _T_ref/40
tf = 5*_T_ref
∂uₜ_∂u = γₜ/(βₜ*dt)
∂uₜₜ_∂u = 1/(βₜ*dt^2)
αₕ = ∂uₜ_∂u/g * (1-βₕ)/βₕ

# Damping (spatial only, no μ dependence)
μ₀ = 2.5
μ₁ᵢₙ(x) = μ₀*(1.0 - sin(π/2*(x[1])/Ld))
μ₁ₒᵤₜ(x) = μ₀*(1.0 - cos(π/2*(x[1]-xdₒᵤₜ)/Ld))

# Parametric space: λ(μ) = μ[1] (wavelength as parameter)
pdomain = (20*h, 30*h)
tdomain = t₀:dt:tf
ptspace = TransientParamSpace(pdomain, tdomain)

λ(μ) = μ[1]
k(μ) = 2π/λ(μ)
ω(μ,t) = x -> √(g*k(μ)*tanh(k(μ)*H))

vᵢₙ(μ,t)  = x -> -(η₀*ω(μ,t)(x)) * cosh(k(μ)*x[2])/sinh(k(μ)*H) * cos(k(μ)*x[1] - ω(μ,t)(x)*t)
vzᵢₙ(μ,t) = x -> ω(μ,t)(x)*η₀*sin(k(μ)*x[1] - ω(μ,t)(x)*t)
ηᵢₙ(μ,t)  = x -> η₀*cos(k(μ)*x[1] - ω(μ,t)(x)*t)

μ₂ᵢₙ(μ,t)  = x -> μ₁ᵢₙ(x)*k(μ)
μ₂ₒᵤₜ(μ,t) = x -> μ₁ₒᵤₜ(x)*k(μ)
ηd(μ,t)    = x -> μ₂ᵢₙ(μ,t)(x)*ηᵢₙ(μ,t)(x)
∇ₙϕd(μ,t) = x -> μ₁ᵢₙ(x)*vzᵢₙ(μ,t)(x)

vμᵢₙ(μ,t)    = parameterise(vᵢₙ,μ,t)
μμ₂ᵢₙ(μ,t)  = parameterise(μ₂ᵢₙ,μ,t)
μμ₂ₒᵤₜ(μ,t) = parameterise(μ₂ₒᵤₜ,μ,t)
ηdμ(μ,t)    = parameterise(ηd,μ,t)
∇ₙϕdμ(μ,t) = parameterise(∇ₙϕd,μ,t)

# Fluid model
domain = (x₀, LΩ, 0.0, H)
partition = (nx_total,ny)
function f_y(x)
  if x == H
      return H
  end
  i = x / (H/ny)
  return H-H/(2.5^i)
end
map_Ω(x) = VectorValue(x[1], f_y(x[2]))
𝒯_Ω = CartesianDiscreteModel(domain,partition,map=map_Ω)

# Labelling
labels_Ω = get_face_labeling(𝒯_Ω)
add_tag_from_tags!(labels_Ω,"surface",[3,4,6])   # assign the label "surface" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[1,2,5])    # assign the label "bottom" to the entity 1,2 and 5 (bottom corners and bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω, "water", [9])       # assign the label "water" to the entity 9 (interior)
# Triangulations
Ω = Interior(𝒯_Ω)
Γ = Boundary(𝒯_Ω,tags="surface")
Γin = Boundary(𝒯_Ω,tags="inlet")

# Auxiliar functions
function is_beam1(xs) # Check if an element is inside the beam1
  n = length(xs)
  x = (1/n)*sum(xs)
  (xb₀ <= x[1] <= xbⱼ ) * ( x[2] ≈ H)
end
function is_beam2(xs) # Check if an element is inside the beam2
  n = length(xs)
  x = (1/n)*sum(xs)
  (xbⱼ <= x[1] <= xb₁ ) * ( x[2] ≈ H)
end
function is_damping1(xs) # Check if an element is inside the damping zone 1
  n = length(xs)
  x = (1/n)*sum(xs)
  (x₀ <= x[1] <= xdᵢₙ ) * ( x[2] ≈ H)
end
function is_damping2(xs) # Check if an element is inside the damping zone 2
  n = length(xs)
  x = (1/n)*sum(xs)
  (xdₒᵤₜ <= x[1] ) * ( x[2] ≈ H)
end
function is_a_joint(xs) # Check if an element is a joint
  is_on_xbⱼ = [x[1]≈xbⱼ && x[2]≈H for x in xs] # array of booleans of size the number of points in an element (for points, it will be an array of size 1)
  element_on_xbⱼ = minimum(is_on_xbⱼ) # Boolean with "true" if at least one entry is true, "false" otherwise.
  element_on_xbⱼ
end

# Beam triangulations
xΓ = get_cell_coordinates(Γ)
Γb1_to_Γ_mask = lazy_map(is_beam1,xΓ)
Γb2_to_Γ_mask = lazy_map(is_beam2,xΓ)
Γd1_to_Γ_mask = lazy_map(is_damping1,xΓ)
Γd2_to_Γ_mask = lazy_map(is_damping2,xΓ)
Γb1_to_Γ = findall(Γb1_to_Γ_mask)
Γb2_to_Γ = findall(Γb2_to_Γ_mask)
Γd1_to_Γ = findall(Γd1_to_Γ_mask)
Γd2_to_Γ = findall(Γd2_to_Γ_mask)
Γf_to_Γ = findall(!,Γb1_to_Γ_mask .| Γb2_to_Γ_mask .| Γd1_to_Γ_mask .| Γd2_to_Γ_mask)
Γη_to_Γ = findall(Γb1_to_Γ_mask .| Γb2_to_Γ_mask )
Γκ_to_Γ = findall(!,Γb1_to_Γ_mask .| Γb2_to_Γ_mask )
Γb1 = Triangulation(Γ,Γb1_to_Γ)
Γb2 = Triangulation(Γ,Γb2_to_Γ)
Γd1 = Triangulation(Γ,Γd1_to_Γ)
Γd2 = Triangulation(Γ,Γd2_to_Γ)
Γfs = Triangulation(Γ,Γf_to_Γ)
Γη = Triangulation(Γ,Γη_to_Γ)
Γκ = Triangulation(Γ,Γκ_to_Γ)
Λb1 = Skeleton(Γb1)
Λb2 = Skeleton(Γb2)

# Construct the mask for the joint
Γ_mask_in_Ω_dim_0 = get_face_mask(labels_Ω,"surface",0)
grid_dim_0_Γ = GridPortion(Grid(ReferenceFE{0},𝒯_Ω),Γ_mask_in_Ω_dim_0)
xΓ_dim_0 = get_cell_coordinates(grid_dim_0_Γ)
Λj_to_Γ_mask = lazy_map(is_a_joint,xΓ_dim_0)
Λj = Skeleton(Γ,Λj_to_Γ_mask)


# Measures
degree = 2*order
dΩ = Measure(Ω,degree)
dΓb1 = Measure(Γb1,degree)
dΓb2 = Measure(Γb2,degree)
dΓd1 = Measure(Γd1,degree)
dΓd2 = Measure(Γd2,degree)
dΓfs = Measure(Γfs,degree)
dΓin = Measure(Γin,degree)
dΛb1 = Measure(Λb1,degree)
dΛb2 = Measure(Λb2,degree)
dΛj = Measure(Λj,degree)

# Normals
nΛb1 = get_normal_vector(Λb1)
nΛb2 = get_normal_vector(Λb2)
nΛj = get_normal_vector(Λj)

# FE spaces
reffe = ReferenceFE(lagrangian,Float64,order)
V_Ω = TestFESpace(Ω, reffe, conformity=:H1)
V_Γκ = TestFESpace(Γκ, reffe, conformity=:H1)
V_Γη = TestFESpace(Γη, reffe, conformity=:H1)
U_Ω  = TransientTrialParamFESpace(V_Ω)
U_Γκ = TransientTrialParamFESpace(V_Γκ)
U_Γη = TransientTrialParamFESpace(V_Γη)
X = TransientMultiFieldFESpace([U_Ω,U_Γκ,U_Γη];style=BlockMultiFieldStyle())
Y = MultiFieldFESpace([V_Ω,V_Γκ,V_Γη];style=BlockMultiFieldStyle())

# Weak form
∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0,1.0)

m(μ,t,(ϕₜₜ,κₜₜ,ηₜₜ),(w,u,v),dΓb1,dΓb2) =
  ∫( d₀*ηₜₜ*v )dΓb1 + ∫( d₀*ηₜₜ*v )dΓb2

c(μ,t,(ϕₜ,κₜ,ηₜ),(w,u,v),dΓfs,dΓd1,dΓd2,dΓb1,dΓb2) =
  ∫( βₕ*ϕₜ*(u + αₕ*w) - κₜ*w )dΓfs +
  ∫( βₕ*ϕₜ*(u + αₕ*w) - κₜ*w )dΓd1 +
  ∫( βₕ*ϕₜ*(u + αₕ*w) - κₜ*w )dΓd2 +
  ∫( ϕₜ*v - ηₜ*w )dΓb1 +
  ∫( ϕₜ*v - ηₜ*w )dΓb2

a(μ,t,(ϕ,κ,η),(w,u,v),dΩ,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) =
  ∫( ∇(w)⋅∇(ϕ) )dΩ +
  ∫( βₕ*(u + αₕ*w)*(g*κ) )dΓfs +
  ∫( βₕ*(u + αₕ*w)*(g*κ) - μμ₂ᵢₙ(μ,t)*κ*w + μ₁ᵢₙ*∇ₙ(ϕ)*(u + αₕ*w) )dΓd1 +
  ∫( βₕ*(u + αₕ*w)*(g*κ) - μμ₂ₒᵤₜ(μ,t)*κ*w + μ₁ₒᵤₜ*∇ₙ(ϕ)*(u + αₕ*w) )dΓd2 +
  ∫( v*(g*η) + a₁*Δ(v)*Δ(η) )dΓb1 +
  ∫( v*(g*η) + a₂*Δ(v)*Δ(η) )dΓb2 +
  ∫( a₁*( - jump(∇(v)⋅nΛb1)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb1) + γ*jump(∇(v)⋅nΛb1)*jump(∇(η)⋅nΛb1) ) )dΛb1 +
  ∫( a₂*( - jump(∇(v)⋅nΛb2)*mean(Δ(η)) - mean(Δ(v))*jump(∇(η)⋅nΛb2) + γ*jump(∇(v)⋅nΛb2)*jump(∇(η)⋅nΛb2) ) )dΛb2 +
  ∫( kᵣ*jump(∇(v)⋅nΛj)*jump(∇(η)⋅nΛj) )dΛj

b(μ,t,(w,u,v),dΓin,dΓd1) =
  ∫( w*vμᵢₙ(μ,t) )dΓin -
  ∫( ηdμ(μ,t)*w - ∇ₙϕdμ(μ,t)*(u + αₕ*w) )dΓd1

res(μ,t,(ϕ,κ,η),(w,u,v),dΩ,dΓin,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) =
  m(μ,t,(∂ₚtt(ϕ),∂ₚtt(κ),∂ₚtt(η)),(w,u,v),dΓb1,dΓb2) +
  c(μ,t,(∂ₚt(ϕ),∂ₚt(κ),∂ₚt(η)),(w,u,v),dΓfs,dΓd1,dΓd2,dΓb1,dΓb2) +
  a(μ,t,(ϕ,κ,η),(w,u,v),dΩ,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) -
  b(μ,t,(w,u,v),dΓin,dΓd1)

trian_res = (Ω,Γin,Γfs,Γd1,Γd2,Γb1,Γb2,Λb1,Λb2,Λj)
trian_m   = (Γb1,Γb2)
trian_c   = (Γfs,Γd1,Γd2,Γb1,Γb2)
trian_a   = (Ω,Γfs,Γd1,Γd2,Γb1,Γb2,Λb1,Λb2,Λj)
domains = FEDomains(trian_res,(trian_a,trian_c,trian_m))

op = TransientLinearParamOperator(res,(a,c,m),ptspace,X,Y,domains)

# Solver
ls = LUSolver()
ode_solver = Newmark(ls,dt,γₜ,βₜ)

# Initial conditions (at rest)
z(μ) = x -> 0.0
zμ(μ) = parameterise(z,μ)
x0(μ) = interpolate_everywhere([zμ(μ),zμ(μ),zμ(μ)],X(μ,t₀))
v0(μ) = interpolate_everywhere([zμ(μ),zμ(μ),zμ(μ)],X(μ,t₀))
a0(μ) = interpolate_everywhere([zμ(μ),zμ(μ),zμ(μ)],X(μ,t₀))

# RB solver
nparams = 10
nparams_res = 10
nparams_jacs = (10,10,10)
dΓκ = Measure(Γκ,degree)
energy((ϕ,κ,η),(w,u,v)) =
  ∫( ∇(ϕ)⋅∇(w) )dΩ +
  ∫(βₕ*g*κ*u)dΓκ + 
  ∫( (∂uₜₜ_∂u*d₀+g)*η*v )dΓb1 + ∫( (∂uₜₜ_∂u*d₀+g)*η*v )dΓb2 +
  ∫( a₁*Δ(v)*Δ(η) )dΓb1 + ∫( a₂*Δ(v)*Δ(η) )dΓb2 +
  ∫( a₁*γ*jump(∇(v)⋅nΛb1)*jump(∇(η)⋅nΛb1) )dΛb1 +
  ∫( a₂*γ*jump(∇(v)⋅nΛb2)*jump(∇(η)⋅nΛb2) )dΛb2

state_reduction = HighDimReduction(1e-4,energy;nparams,sketch=:sprn)
rbsolver = RBSolver(ode_solver,state_reduction;nparams_res,nparams_jacs)

test_case = get(ENV,"TEST_CASE","")
final_case = get(ENV,"FINAL_CASE","")
@assert !isempty(test_case) && !isempty(final_case) "TEST_CASE and FINAL_CASE environment variables are required"

i = parse(Int,test_case)
ifin = parse(Int,final_case)

start = (i-1)*nparams+1
μ = i == ifin ? realisation(op;nparams,sampling=:uniform) : realisation(op;nparams,start)

problem = Problem(rbsolver,op,μ,"khaba_time_ord2_$(i)",(x₀,v₀,a₀))
run_problem(problem)