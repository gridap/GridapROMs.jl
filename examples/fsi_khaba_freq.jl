using DrWatson
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using GridapROMs
using GridapROMs.Utils 
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures

include("ExamplesInterface.jl")

# Fixed parameters
Lb = 12.5
m = 8.36
EI₁ = 47100.0
EI₂(μ) = EI₁ / μ[3]
β = 0.2
H = 1.1
α = 0.249

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
ρ = 1025
d₀ = m/ρ
a₁ = EI₁/ρ
a₂(μ) = x -> EI₂(μ)/ρ
ξ(μ) = μ[2]
kᵣ(μ) = x -> ξ(μ)*a₁/Lb

# Numerics constants
nx = 20
ny = 5
nx_total = Int(ceil(nx/β)*ceil(LΩ/Lb))
h = LΩ / nx_total
order = 2
γ = 1.0*order*(order-1)/h
βₕ = 0.5

# Damping
μ₀ = 2.5
μ₁ᵢₙ(x) = μ₀*(1.0 - sin(π/2*(x[1])/Ld))
μ₁ₒᵤₜ(x) = μ₀*(1.0 - cos(π/2*(x[1]-xdₒᵤₜ)/Ld))

# (Parametric) wave/material properties
pdomain = (20*h,30*h,0.0,625.0,1.0,500.0)
pspace = ParamSpace(pdomain)

η₀ = 0.01 
λ(μ) = μ[1] # α*Lb
k(μ) = 2π/λ(μ)
ω(μ) = x -> √(g*k(μ)*tanh(k(μ)*H))
ηᵢₙ(μ) = x -> η₀*exp(im*k(μ)*x[1])
vᵢₙ(μ) = x -> (η₀*ω(μ)(x))*(cosh(k(μ)*x[2]) / sinh(k(μ)*H))*exp(im*k(μ)*x[1])
vzᵢₙ(μ) = x -> -im*ω(μ)(x)*η₀*exp(im*k(μ)*x[1])
αₕ(μ) = x -> -im*ω(μ)(x)/g * (1-βₕ)/βₕ
μ₂ᵢₙ(μ) = x -> μ₁ᵢₙ(x)*k(μ)
μ₂ₒᵤₜ(μ) = x -> μ₁ₒᵤₜ(x)*k(μ)
ηd(μ) = x -> μ₂ᵢₙ(μ)(x)*ηᵢₙ(μ)(x)
∇ₙϕd(μ) = x -> μ₁ᵢₙ(x)*vzᵢₙ(μ)(x)

a₂μ(μ) = parameterise(a₂,μ)
kᵣμ(μ) = parameterise(kᵣ,μ)
ωμ(μ) = parameterise(ω,μ)
vμᵢₙ(μ) = parameterise(vᵢₙ,μ)
αμₕ(μ) = parameterise(αₕ,μ)
μμ₂ᵢₙ(μ) = parameterise(μ₂ᵢₙ,μ)
μμ₂ₒᵤₜ(μ) = parameterise(μ₂ₒᵤₜ,μ)
ηdμ(μ) = parameterise(ηd,μ)
∇ₙϕdμ(μ) = parameterise(∇ₙϕd,μ)

# Fluid model
domain = (x₀,LΩ,0.0,H)
partition = (nx_total,ny)
function f_y(x)
  if x == H
      return H
  end
  i = x / (H/ny)
  return H-H/(2.5^i)
end
map_Ω(x) = VectorValue(x[1],f_y(x[2]))
𝒯_Ω = CartesianDiscreteModel(domain,partition,map=map_Ω)

# Labelling
labels_Ω = get_face_labeling(𝒯_Ω)
add_tag_from_tags!(labels_Ω,"surface",[3,4,6])   # assign the label "surface" to the entity 3,4 and 6 (top corners and top side)
add_tag_from_tags!(labels_Ω,"bottom",[1,2,5])    # assign the label "bottom" to the entity 1,2 and 5 (bottom corners and bottom side)
add_tag_from_tags!(labels_Ω,"inlet",[7])         # assign the label "inlet" to the entity 7 (left side)
add_tag_from_tags!(labels_Ω,"outlet",[8])        # assign the label "outlet" to the entity 8 (right side)
add_tag_from_tags!(labels_Ω,"water",[9])       # assign the label "water" to the entity 9 (interior)

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

function is_a_joint(xs) 
  is_on_xbⱼ = [x[1]≈xbⱼ && x[2]≈H for x in xs] 
  element_on_xbⱼ = minimum(is_on_xbⱼ) 
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
V_Ω = TestFESpace(Ω,reffe,conformity=:H1,vector_type=Vector{ComplexF64})
V_Γκ = TestFESpace(Γκ,reffe,conformity=:H1,vector_type=Vector{ComplexF64})
V_Γη = TestFESpace(Γη,reffe,conformity=:H1,vector_type=Vector{ComplexF64})
U_Ω = ParamTrialFESpace(V_Ω)
U_Γκ = ParamTrialFESpace(V_Γκ)
U_Γη = ParamTrialFESpace(V_Γη)
# X = MultiFieldFESpace([U_Ω,U_Γκ,U_Γη];style=BlockMultiFieldStyle())
# Y = MultiFieldFESpace([V_Ω,V_Γκ,V_Γη];style=BlockMultiFieldStyle())
X = MultiFieldFESpace([U_Ω,U_Γκ,U_Γη])
Y = MultiFieldFESpace([V_Ω,V_Γκ,V_Γη])

# Weak form
∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0,1.0)
a(μ,(ϕ,κ,η),(w,u,v),dΩ,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) =      
  ∫(  ∇(w)⋅∇(ϕ) )dΩ   +
  ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ )dΓfs   +
  ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ - μμ₂ᵢₙ(μ)*κ*w + μ₁ᵢₙ*∇ₙ(ϕ)*(u + αμₕ(μ)*w) )dΓd1    +
  ∫(  βₕ*(u + αμₕ(μ)*w)*(g*κ - im*ωμ(μ)*ϕ) + im*ωμ(μ)*w*κ - μμ₂ₒᵤₜ(μ)*κ*w + μ₁ₒᵤₜ*∇ₙ(ϕ)*(u + αμₕ(μ)*w) )dΓd2    +
  ∫(  ( v*((-ωμ(μ)^2*d₀ + g)*η - im*ωμ(μ)*ϕ) + a₁*Δ(v)*Δ(η) ) +  im*ωμ(μ)*w*η  )dΓb1  +
  ∫(  ( v*((-ωμ(μ)^2*d₀ + g)*η - im*ωμ(μ)*ϕ) + a₂μ(μ)*Δ(v)*Δ(η) ) +  im*ωμ(μ)*w*η  )dΓb2  +
  ∫(  a₁ * ( - jump(∇(v)⋅nΛb1) * mean(Δ(η)) - mean(Δ(v)) * jump(∇(η)⋅nΛb1) + γ*( jump(∇(v)⋅nΛb1) * jump(∇(η)⋅nΛb1) ) ) )dΛb1 +
  ∫(  a₂μ(μ) * ( - jump(∇(v)⋅nΛb2) * mean(Δ(η)) - mean(Δ(v)) * jump(∇(η)⋅nΛb2) + γ*( jump(∇(v)⋅nΛb2) * jump(∇(η)⋅nΛb2) ) ) )dΛb2 +
  ∫(  (jump(∇(v)⋅nΛj) * kᵣμ(μ) * jump(∇(η)⋅nΛj)) )dΛj
l(μ,(w,u,v),dΓin,dΓd1) = ∫( w*vμᵢₙ(μ) )dΓin - ∫( ηdμ(μ)*w - ∇ₙϕdμ(μ)*(u + αμₕ(μ)*w) )dΓd1
res(μ,ϕ,w,dΩ,dΓin,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) =
  a(μ,ϕ,w,dΩ,dΓfs,dΓd1,dΓd2,dΓb1,dΓb2,dΛb1,dΛb2,dΛj) - l(μ,w,dΓin,dΓd1)

trian_res = (Ω,Γin,Γfs,Γd1,Γd2,Γb1,Γb2,Λb1,Λb2,Λj)
trian_a = (Ω,Γfs,Γd1,Γd2,Γb1,Γb2,Λb1,Λb2,Λj)
domains = FEDomains(trian_res,trian_a)

op = LinearParamOperator(res,a,pspace,X,Y,domains)

# RB solver 
solver = LUSolver()
tol = 1e-4
nparams = 90
nparams_res = 90
nparams_jac = 90

_EI₂ = 500.0
_a₂ = _EI₂/ρ
_ξ = 625.0
_kᵣ = _ξ*a₁/Lb
_λ = 20*h 
_k = 2pi/_λ
_ω = √(g*_k*tanh(_k*H))

dΓκ = Measure(Γκ,degree)
energy((ϕ,κ,η),(w,u,v)) = 
  ∫(∇(ϕ)⋅∇(w))dΩ + 
  ∫(βₕ*g*κ*u)dΓκ + 
  ∫((_ω^2*d₀+g)*η*v)dΓb1 + ∫((_ω^2*d₀+g)*η*v)dΓb2 +
  ∫(a₁*Δ(v)*Δ(η))dΓb1 + ∫(_a₂*Δ(v)*Δ(η))dΓb2 +
  ∫(a₁*γ*( jump(∇(v)⋅nΛb1) * jump(∇(η)⋅nΛb1) ) )dΛb1 + 
  ∫(_a₂*γ*( jump(∇(v)⋅nΛb2) * jump(∇(η)⋅nΛb2) ) )dΛb2 + 
  ∫(_kᵣ*(jump(∇(v)⋅nΛj)*jump(∇(η)⋅nΛj)))dΛj
state_reduction = PODReduction(tol,energy;nparams,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction;nparams_res,nparams_jac)

dir = datadir("khaba_freq")
create_dir(dir)

fesnaps,festats = solution_snapshots(rbsolver,op)
save(dir,fesnaps)
save(dir,festats)

μ = realisation(pspace;nparams=10,sampling=:uniform)
x,stats = solution_snapshots(rbsolver,op,μ)
save(dir,x;label=online_label)
save(dir,stats;label=online_label)
# tols = [1e-1,1e-2,1e-3,1e-4,1e-5]
# run_test(
#   dir,
#   rbsolver,
#   op,
#   tols;
#   nparams=10,
# )