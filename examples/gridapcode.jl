using DrWatson
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using BenchmarkTools

# Fixed parameters
Lb = 12.5
m = 8.36
EI₁ = 47100.0
EI₂ = 471.0
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
a₂ = EI₂/ρ
ξ = 0.0
kᵣ = ξ*a₁/Lb

# Numerics constants
nx = 20
ny = 5
nx_total = Int(ceil(nx/β)*ceil(LΩ/Lb))
h = LΩ / nx_total
order = 4
γ = 1.0*order*(order-1)/h
βₕ = 0.5

# Damping
μ₀ = 2.5
μ₁ᵢₙ(x) = μ₀*(1.0 - sin(π/2*(x[1])/Ld))
μ₁ₒᵤₜ(x) = μ₀*(1.0 - cos(π/2*(x[1]-xdₒᵤₜ)/Ld))

η₀ = 0.01 
λ = α*Lb
k = 2π/λ
ω = √(g*k*tanh(k*H))
ηᵢₙ(x) = η₀*exp(im*k*x[1])
vᵢₙ(x) = (η₀*ω)*(cosh(k*x[2]) / sinh(k*H))*exp(im*k*x[1])
vzᵢₙ(x) = -im*ω*η₀*exp(im*k*x[1])
αₕ = -im*ω/g * (1-βₕ)/βₕ
μ₂ᵢₙ(x) = μ₁ᵢₙ(x)*k
μ₂ₒᵤₜ(x) = μ₁ₒᵤₜ(x)*k
ηd(x) = μ₂ᵢₙ(x)*ηᵢₙ(x)
∇ₙϕd(x) = μ₁ᵢₙ(x)*vzᵢₙ(x)

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
U_Ω = TrialFESpace(V_Ω)
U_Γκ = TrialFESpace(V_Γκ)
U_Γη = TrialFESpace(V_Γη)
X = MultiFieldFESpace([U_Ω,U_Γκ,U_Γη];style=BlockMultiFieldStyle())
Y = MultiFieldFESpace([V_Ω,V_Γκ,V_Γη];style=BlockMultiFieldStyle())

Xc = MultiFieldFESpace([U_Ω,U_Γκ,U_Γη])
Yc = MultiFieldFESpace([V_Ω,V_Γκ,V_Γη])

# Weak form
∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0,1.0)
a((ϕ,κ,η),(w,u,v)) =  
  ∫(  ∇(w)⋅∇(ϕ) )dΩ   +
  ∫(  βₕ*(u + αₕ*w)*(g*κ - im*ω*ϕ) + im*ω*w*κ )dΓfs   +
  ∫(  βₕ*(u + αₕ*w)*(g*κ - im*ω*ϕ) + im*ω*w*κ - μ₂ᵢₙ*κ*w + μ₁ᵢₙ*∇ₙ(ϕ)*(u + αₕ*w) )dΓd1    +
  ∫(  βₕ*(u + αₕ*w)*(g*κ - im*ω*ϕ) + im*ω*w*κ - μ₂ₒᵤₜ*κ*w + μ₁ₒᵤₜ*∇ₙ(ϕ)*(u + αₕ*w) )dΓd2    +
  ∫(  ( v*((-ω^2*d₀ + g)*η - im*ω*ϕ) + a₁*Δ(v)*Δ(η) ) +  im*ω*w*η  )dΓb1  +
  ∫(  ( v*((-ω^2*d₀ + g)*η - im*ω*ϕ) + a₂*Δ(v)*Δ(η) ) +  im*ω*w*η  )dΓb2  +
  ∫(  a₁ * ( - jump(∇(v)⋅nΛb1) * mean(Δ(η)) - mean(Δ(v)) * jump(∇(η)⋅nΛb1) + γ*( jump(∇(v)⋅nΛb1) * jump(∇(η)⋅nΛb1) ) ) )dΛb1 +
  ∫(  a₂ * ( - jump(∇(v)⋅nΛb2) * mean(Δ(η)) - mean(Δ(v)) * jump(∇(η)⋅nΛb2) + γ*( jump(∇(v)⋅nΛb2) * jump(∇(η)⋅nΛb2) ) ) )dΛb2 +
  ∫(  (jump(∇(v)⋅nΛj) * kᵣ * jump(∇(η)⋅nΛj)) )dΛj

l((w,u,v)) = ∫( w*vᵢₙ )dΓin - ∫( ηd*w - ∇ₙϕd*(u + αₕ*w) )dΓd1

dx,y = get_trial_fe_basis(X),get_fe_basis(Y)
@btime a(dx,y)
@btime l(y)

dca = a(dx,y)
matcontribs = collect_cell_matrix(X,Y,dca)
assem = SparseMatrixAssembler(X,Y)
A = assemble_matrix(assem,matcontribs)
@btime assemble_matrix(assem,matcontribs)

assemc = SparseMatrixAssembler(Xc,Yc)
@btime assemble_matrix(assemc,matcontribs)

op = AffineFEOperator(a,l,X,Y)
@btime solve($op);

opc = AffineFEOperator(a,l,Xc,Yc)
@btime solve($opc);

# gridaproms 

_λ(μ) = μ[1] 
_k(μ) = 2π/λ(μ)
_ω(μ) = x -> √(g*_k(μ)*tanh(_k(μ)*H))
_ηᵢₙ(μ) = x -> η₀*exp(im*_k(μ)*x[1])
_vᵢₙ(μ) = x -> (η₀*_ω(μ)(x))*(cosh(_k(μ)*x[2]) / sinh(_k(μ)*H))*exp(im*_k(μ)*x[1])
_vzᵢₙ(μ) = x -> -im*_ω(μ)(x)*η₀*exp(im*_k(μ)*x[1])
_αₕ(μ) = x -> -im*_ω(μ)(x)/g * (1-βₕ)/βₕ
_μ₂ᵢₙ(μ) = x -> μ₁ᵢₙ(x)*_k(μ)
_μ₂ₒᵤₜ(μ) = x -> μ₁ₒᵤₜ(x)*_k(μ)
_ηd(μ) = x -> _μ₂ᵢₙ(μ)(x)*_ηᵢₙ(μ)(x)
_∇ₙϕd(μ) = x -> _μ₁ᵢₙ(x)*_vzᵢₙ(μ)(x)

ωμ(μ) = parameterise(_ω,μ)
vμᵢₙ(μ) = parameterise(_vᵢₙ,μ)
αμₕ(μ) = parameterise(_αₕ,μ)
μμ₂ᵢₙ(μ) = parameterise(_μ₂ᵢₙ,μ)
μμ₂ₒᵤₜ(μ) = parameterise(_μ₂ₒᵤₜ,μ)
ηdμ(μ) = parameterise(_ηd,μ)
∇ₙϕdμ(μ) = parameterise(_∇ₙϕd,μ)

_U_Ω = ParamTrialFESpace(V_Ω)
_U_Γκ = ParamTrialFESpace(V_Γκ)
_U_Γη = ParamTrialFESpace(V_Γη)
_X = MultiFieldFESpace([_U_Ω,_U_Γκ,_U_Γη];style=BlockMultiFieldStyle())
_Y = MultiFieldFESpace([_V_Ω,_V_Γκ,_V_Γη];style=BlockMultiFieldStyle())