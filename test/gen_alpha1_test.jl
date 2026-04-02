using Test

using LinearAlgebra

using Gridap
using Gridap.Algebra
using Gridap.FESpaces
using Gridap.ODEs

using BlockArrays

# Geometry
domain = (0,1,0,1)
partition = (5,5)
model = CartesianDiscreteModel(domain,partition)

# FE spaces
udt(t) = x -> 0.0
ud = TimeSpaceFunction(udt)
order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V = FESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
U = TransientTrialFESpace(V,ud)

# Integration
Ω = Triangulation(model)
degree = 2 * order
dΩ = Measure(Ω,degree)

# FE operator
ft(t) = x -> t
f = TimeSpaceFunction(ft)
mass(t,∂ₜu,v) = ∫(∂ₜu ⋅ v) * dΩ
stiffness(t,u,v) = ∫(∇(u) ⊙ ∇(v)) * dΩ
forcing(t,v) = ∫(f(t) ⋅ v) * dΩ

tfeop = TransientLinearFEOperator(stiffness,mass,forcing,U,V)

# Initial conditions
t0 = 0.0
tF = 0.3
dt = 0.1

uf0(x) = 1.0
u̇f0(x) = 1.0
U0 = U(t0)
uh0 = interpolate_everywhere(uf0,U0)
u̇h0 = interpolate_everywhere(u̇f0,U0)
uhs0 = (uh0,u̇h0)

sysslvr_l = LUSolver()
odeslvr = GeneralizedAlpha1(LUSolver(), dt, 0.5)
fesltn = solve(odeslvr,tfeop,t0,tF,uhs0)

Ns = num_free_dofs(V)
Nt = round(Int,(tF - t0)/dt)
Uh = zeros(Ns,Nt)
@views for (i,(t_n,uh_n)) in enumerate(fesltn)
  Uh[:,i] = get_free_dof_values(uh_n)
end

αf,αm,γ = odeslvr.αf,odeslvr.αm,odeslvr.γ 

a = 1 / (γ*dt)
b = 1 - 1/γ
c = a * ( (1-αm) + αm*b )

A = assemble_matrix((u,v) -> stiffness(0.0,u,v),U0,V)
M = assemble_matrix((u,v) -> mass(0.0,u,v),U0,V)

t1 = t0 + dt
t2 = t1 + dt
t3 = t2 + dt
tα1 = (1 - αf)*t0 + αf*t1
tα2 = (1 - αf)*t1 + αf*t2
tα3 = (1 - αf)*t2 + αf*t3

u0 = get_free_dof_values(uh0)
u̇0 = get_free_dof_values(u̇h0)

_fα1 = assemble_vector(v -> forcing(tα1,v),V)
_fα2 = assemble_vector(v -> forcing(tα2,v),V)
_fα3 = assemble_vector(v -> forcing(tα3,v),V)

B11 = a * αm * M + αf * A
B21 = - a * αm * M + (1 - αf) * A#(c - a * αm) * M + (1 - αf) * A
Budot = c / a * M # ((1 - αm) + αm * b) * M 
f1 = _fα1 - B21 * u0 - Budot * u̇0 
@assert B11 * Uh[:,1] ≈ f1

B22 = B11 
B21 = (c - a * αm) * M + (1 - αf) * A
f2 = _fα2 - c / a * b^1 * M * u̇0 + c * b^0 * M * u0 
@assert B22 * Uh[:,2] + B21 * Uh[:,1] ≈ f2

B33 = B11 
B32 = B21
B31 = c * (b^1 - b^0) * M
f3 = _fα3 - c / a * b^2 * M * u̇0 + c * b^1 * M * u0 
@assert B33 * Uh[:,3] + B32 * Uh[:,2] + B31 * Uh[:,1] ≈ f3


_B = Matrix{Matrix{Float64}}(undef,Nt,Nt)
_B[1,1] = B11
_B[2,2] = B11
_B[3,3] = B11
_B[2,1] = B21
_B[3,2] = B32
_B[3,1] = B31
_B[1,2] = zeros(size(M))
_B[1,3] = zeros(size(M))
_B[2,3] = zeros(size(M))

_F = Vector{Vector{Float64}}(undef,Nt)
_F[1] = _fα1 - c / a * b^0 * M * u̇0 - Bn(1) * M * u0 - _B[2,1] * u0
_F[2] = _fα2 - c / a * b^1 * M * u̇0 - Bn(2) * M * u0 
_F[3] = _fα3 - c / a * b^2 * M * u̇0 - Bn(3) * M * u0