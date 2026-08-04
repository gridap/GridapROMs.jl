using Gridap
using Gridap.ODEs
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs

println("=== TRANSIENT DEEPONET TEST ===")

# ===================================================================
# 1. HIGH-FIDELITY FEM SETUP
# ===================================================================
# Geometry
Ω = (0.0,1.0)
parts = (10,)
Ωₕ = CartesianDiscreteModel(Ω,parts)
τₕ = Triangulation(Ωₕ)

# Temporal grid
θ = 0.5 # Crank-Nicolson
dt = 0.01
t0 = 0.0
tf = 10 * dt
tdomain = t0:dt:tf

# Parametric quantities
pdomain = (1.0,5.0)
D = TransientParamSpace(pdomain,tdomain)

# Manufactured solution and source term
u(μ,t) = x -> t * (μ[1] * x[1]^2)
uₚₜ(μ,t) = parameterise(u,μ,t)

# Analytical Laplacian of u: Δ(t * μ * x^2) = 2 * t * μ
f(μ,t) = x -> -2.0 * t * μ[1]
fₚₜ(μ,t) = parameterise(f,μ,t)

# Numerical integration
order = 1
dΩₕ = Measure(τₕ,2 * order)

# Weak form
a(μ,t,du,v,dΩ) = ∫(∇(v) ⋅ ∇(du))dΩ
m(μ,t,du,v,dΩ) = ∫(v * du)dΩ
r(μ,t,u,v,dΩ) = m(μ,t,∂t(u),v,dΩ) + a(μ,t,u,v,dΩ) - ∫(fₚₜ(μ,t) * v)dΩ

# Domains
τₕ_a = (τₕ,)
τₕ_m = (τₕ,)
τₕ_r = (τₕ,)
domains = FEDomains(τₕ_r,(τₕ_a,τₕ_m))

# FE Spaces
reffe = ReferenceFE(lagrangian,Float64,order)
V_standard = TestFESpace(Ωₕ,reffe; dirichlet_tags="boundary")

V = OrderedFESpace(V_standard)
U = TransientTrialParamFESpace(V,uₚₜ)

# Operator
feop = TransientLinearParamOperator(r,(a,m),D,U,V,domains)

# Initial condition
u₀(μ) = x -> 0.0
u₀ₚ(μ) = parameterise(u₀,μ)
uh₀ₚ(μ) = interpolate_everywhere(u₀ₚ(μ),U(μ,t0))

# FE solver
fe_solver = ThetaMethod(LUSolver(),dt,θ)

# Generate Snapshots
println("Generating High-Fidelity Transient Snapshots...")
μₒₙ = realisation(D; nparams=5,sampling=:uniform)
s,_ = solution_snapshots(fe_solver,feop,μₒₙ,uh₀ₚ)
println("Snapshots generated. Shape: ",size(get_all_data(s)))

# ===================================================================
# 2. NEURAL OPERATOR SETUP
# ===================================================================
model = DeepONet(
  branch_layers = (1, 32, 32, 16),
  trunk_layers  = (2, 32, 32, 16), # <-- Starts with 2 (x, t)
  activation    = tanh
)

strategy = DeepONetStrategy(
  model = model,
  epochs = 10,
  batch_size = 0,
  step_x = 1,
  step_t = 1,
  m_sensors = 1
)

reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(fe_solver,reduction)

# ===================================================================
# 3. OFFLINE PHASE (TRAINING)
# ===================================================================
println("\nStarting Transient Offline Phase...")

# This will trigger our RBTransient.reduced_operator dispatch and
# then the RBTransient.train_neural_operator function we wrote!
neural_rb_op = reduced_operator(solver,feop,s)

println("Transient Offline phase completed successfully!")