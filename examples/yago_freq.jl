using DrWatson
using Gridap
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.CellData
using Gridap.MultiField
using Gridap.TensorValues
using Gridap.ODEs
using GridapROMs
using GridapROMs.Utils 
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapSolvers
using GridapSolvers.LinearSolvers

# Fixed parameters
L = 300
B = 60
H = 58.5
hb = 2.0
nLΩ = 10
nBΩ = 18
LΩ = nLΩ*L
BΩ = nBΩ*B
xb₀ = 4.5*L
xb₁ = xb₀ + L
yb₀ = -B/2
yb₁ = yb₀ + B

## Numerics (space discretization)
nx = 32
ny = 4
nz = 4
order = 2
h = LΩ/(nLΩ*nx)
βₕ = 0.5
γ = 1.0*order*(order+1)/h

## Physics
g = 9.81
ρ(μ) = μ[3]
E(μ) = 11.9e9 / μ[3]
ρb = 256.25
ν = 0.13
I_b = hb^3/12
d₀(μ) = x -> ρb*hb/ρ(μ)

δ(x,y) = ==(x,y)
C_type = SymFourthOrderTensorValue{3,Float64}
μ_s = 1.0/(2*(1+ν))   # Lamé shear modulus divided by E
λ_s = ν/(1-ν^2)  # Lamé first parameter divided by E
Cvals = zero(Array{Float64}(undef,36))
for ii in 1:2
  for jj in 1:2
    for kk in 1:2
      for ll in 1:2
        Cvals[data_index(C_type,ii,jj,kk,ll)] = I_b*(μ_s*(δ(ii,kk)*δ(jj,ll) + δ(ii,ll)*δ(jj,kk)) + λ_s*(δ(ii,jj)*δ(kk,ll)))
      end
    end
  end
end
Ĉ = SymFourthOrderTensorValue(Cvals...)

# Damping
μ₀ = 2.5
dfactor = 3
Ld = dfactor*L
Ld₀ = dfactor*L
xd = LΩ - Ld
xd₀ = Ld₀
μ₁(x) = μ₀*(1.0 - cos(π/2*(x[1]-xd)/Ld)) * (x[1]>xd) + μ₀*(1-cos(π/2*(Ld₀-x[1])/Ld₀)) * (x[1]<xd₀)

# Parametric wave/material properties
pdomain = (20*h,30*h,900.0,1100.0,1.0,500.0)
pspace = ParamSpace(pdomain)

η₀ = 0.01
λ(μ) = μ[1]
k(μ) = 2π/λ(μ)
ω(μ) = x -> √(g*k(μ)*tanh(k(μ)*H))
αₕ(μ) = x -> -im*ω(μ)(x)/g * (1-βₕ)/βₕ
ηᵢₙ(μ) = x -> η₀*exp(im*k(μ)*x[1])
vᵢₙ(μ) = x -> (η₀*ω(μ)(x))*(cosh(k(μ)*x[3]) / sinh(k(μ)*H))*exp(im*k(μ)*x[1])
vzᵢₙ(μ) = x -> -im*ω(μ)(x)*η₀*exp(im*k(μ)*x[1])
μ₂(μ) = x -> μ₁(x)*k(μ)
D(μ) = E(μ)*I_b/(1.0-ν^2)
Dᵨ(μ) = x -> D(μ)/ρ(μ)
ηd(μ) = x -> μ₂(μ)(x)*ηᵢₙ(μ)(x)*(x[1]<xd₀)
∇ₙϕd(μ) = x -> μ₁(x)*vzᵢₙ(μ)(x)*(x[1]<xd₀)

ωμ(μ) = parameterise(ω,μ)
αμₕ(μ) = parameterise(αₕ,μ)
vμᵢₙ(μ) = parameterise(vᵢₙ,μ)
μ₂μ(μ) = parameterise(μ₂,μ)
Dᵨμ(μ) = parameterise(Dᵨ,μ)
d₀μ(μ) = parameterise(d₀,μ)
ηdμ(μ) = parameterise(ηd,μ)
∇ₙϕdμ(μ) = parameterise(∇ₙϕd,μ)

# Define fluid model
domain = (0.0,LΩ,-BΩ/2,BΩ/2,0.0,H)
partition = (nLΩ*nx,nBΩ*ny,nz)
function f_z(x)
  if x == H
      return H
  end
  i = x / (H/nz)
  return H-H/((2.5)^i)
end
map_Ω(x) = VectorValue(x[1],x[2],f_z(x[3]))
model_Ω = CartesianDiscreteModel(domain,partition,map=map_Ω)

# Add labels to Ω
labels_Ω = get_face_labeling(model_Ω)
add_tag_from_tags!(labels_Ω,"surface",[22])
add_tag_from_tags!(labels_Ω,"inlet",[25])

# Triangulations
Ω = Interior(model_Ω)
Γ = Boundary(model_Ω,tags="surface")
Γᵢₙ = Boundary(model_Ω,tags="inlet")

# Create masks in Γ
function is_plate(x)
  is_in = ([(xb₀ <= xm[1]) * (xm[1] <= xb₁) * (yb₀ <= xm[2]) * (xm[2] <= yb₁) for xm in x])
  minimum(is_in)
end
xΓ = get_cell_coordinates(Γ)
Γb_to_Γ_mask = lazy_map(is_plate,xΓ)
Γb_to_Γ = findall(Γb_to_Γ_mask)
Γf_to_Γ = findall(!,Γb_to_Γ_mask)
Γb = Triangulation(Γ,Γb_to_Γ)
Γf = Triangulation(Γ,Γf_to_Γ)
Λb = Skeleton(Γb)

# Measures
degree = 2*order
dΩ = Measure(Ω,degree)
dΓb = Measure(Γb,degree)
dΓf = Measure(Γf,degree)
dΓᵢₙ = Measure(Γᵢₙ,degree)
dΛb = Measure(Λb,degree)

# Normals
nΛb = get_normal_vector(Λb)

# FE spaces
reffeη = ReferenceFE(lagrangian,Float64,order)
reffeκ = ReferenceFE(lagrangian,Float64,order)
reffeᵩ = ReferenceFE(lagrangian,Float64,2)
V_Ω  = TestFESpace(Ω,reffeᵩ,conformity=:H1,vector_type=Vector{ComplexF64})
V_Γf = TestFESpace(Γf,reffeκ,conformity=:H1,vector_type=Vector{ComplexF64})
V_Γb = TestFESpace(Γb,reffeη,conformity=:H1,vector_type=Vector{ComplexF64})
U_Ω  = ParamTrialFESpace(V_Ω)
U_Γf = ParamTrialFESpace(V_Γf)
U_Γb = ParamTrialFESpace(V_Γb)
X = MultiFieldFESpace([U_Ω,U_Γf,U_Γb])
Y = MultiFieldFESpace([V_Ω,V_Γf,V_Γb])

# Weak form
∇ₙ(ϕ) = ∇(ϕ)⋅VectorValue(0.0,0.0,1.0)
a(μ,(ϕ,κ,η),(w,u,v),dΩ,dΓf,dΓb,dΛb) =
  ∫( ∇(ϕ)⋅∇(w) )dΩ +
  ∫( βₕ*(g*κ - im*ωμ(μ)*ϕ)*(u + αμₕ(μ)*w) + im*ωμ(μ)*κ*w - μ₂μ(μ)*κ*w + μ₁*∇ₙ(ϕ)*(u + αμₕ(μ)*w) )dΓf +
  ∫( ((g - ωμ(μ)^2*d₀μ(μ))*η - im*ωμ(μ)*ϕ)*v + im*ωμ(μ)*η*w + (∇∇(v)⊙(Ĉ⊙∇∇(η))) )dΓb +
  ∫( ( - jump(∇(v))⊙(mean(Ĉ⊙∇∇(η))⋅nΛb.⁺) - (mean(Ĉ⊙∇∇(v))⋅nΛb.⁺)⊙jump(∇(η)) + jump(Dᵨμ(μ)*γ*∇(v))⊙jump(∇(η)) ) )dΛb

l(μ,(w,u,v),dΓᵢₙ,dΓf) =
  ∫( vμᵢₙ(μ)*w )dΓᵢₙ - ∫( ηdμ(μ)*w - ∇ₙϕdμ(μ)*(u + αμₕ(μ)*w) )dΓf

res(μ,(ϕ,κ,η),(w,u,v),dΩ,dΓᵢₙ,dΓf,dΓb,dΛb) =
  a(μ,(ϕ,κ,η),(w,u,v),dΩ,dΓf,dΓb,dΛb) -
  l(μ,(w,u,v),dΓᵢₙ,dΓf)

trian_res = (Ω,Γᵢₙ,Γf,Γb,Λb)
trian_a = (Ω,Γf,Γb,Λb)
domains = FEDomains(trian_res,trian_a)

op = LinearParamOperator(res,a,pspace,X,Y,domains)

# RB solver
solver = LUSolver()
tol = 1e-4
nparams = 50
nparams_res = 50
nparams_jac = 50
_μ_ref = (20*h,900.0,500.0)
_E = E(_μ_ref)
_ρ = ρ(_μ_ref)
_D = _E*I_b/(1.0-ν^2)
_Dᵨ = _D/_ρ
_C = (_E/_ρ)*Ĉ
_λ = _μ_ref[1]
_k = 2π/_λ
_ω = √(g*_k*tanh(_k*H))
_d₀ = ρb*hb/ρ(_μ_ref)

energy((ϕ,κ,η),(w,u,v)) =
  ∫( ∇(ϕ)⋅∇(w) )dΩ +
  ∫( βₕ*g*κ*u + ∇(κ)⋅∇(u) )dΓf +
  ∫( (_ω^2*_d₀+g)*η*v + ∇∇(v)⊙(_C⊙∇∇(η)) )dΓb +
  ∫( _Dᵨ*γ*jump(∇(v))⊙jump(∇(η)) )dΛb

state_reduction = PODReduction(tol,energy; nparams,sketch=:sprn)
rbsolver = RBSolver(solver,state_reduction; nparams_res,nparams_jac)

# Offline + online phases
dir = datadir("yago_freq")
create_dir(dir)

fesnaps,= solution_snapshots(rbsolver,op)
rbop = reduced_operator(rbsolver,op,fesnaps)
μon = realisation(op; nparams=10,sampling=:uniform)
x̂,rbstats = solve(rbsolver,rbop,μon)
x,festats = solution_snapshots(rbsolver,op,μon)
perf = eval_performance(rbsolver,rbop,x,x̂,festats,rbstats)
