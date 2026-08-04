using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamSteady

println("=== STEADY DEEPONET TEST ===")

# ===================================================================
# 1. HIGH-FIDELITY FEM SETUP
# ===================================================================
ν(μ) = x -> 1.0 + μ[1]
f(μ) = x -> 1.0
g(μ) = x -> x[1]

νₚ(μ) = parameterise(ν,μ)
fₚ(μ) = parameterise(f,μ)
gₚ(μ) = parameterise(g,μ)

Ω = (0.0,1.0)
partition = (20,)
Ωₕ = CartesianDiscreteModel(Ω,partition)

order = 2
reffe = ReferenceFE(lagrangian,Float64,order)
V_standard = TestFESpace(Ωₕ,reffe; dirichlet_tags=[1,2])

V = OrderedFESpace(V_standard)
U = ParamTrialFESpace(V,gₚ)

D = ParamSpace((1e-3,10.0),sampling=:halton)
μₒₙ = realisation(D; nparams=10,sampling=:uniform)

degree = 2 * order
τₕ = Triangulation(Ωₕ)
dΩₕ = Measure(τₕ,degree)

a(μ,u,v,dΩ) = ∫(νₚ(μ) * ∇(v) ⋅ ∇(u))dΩ
r(μ,u,v,dΩ) = a(μ,u,v,dΩ) - ∫(fₚ(μ) * v)dΩ

τₕ_r = (τₕ,)
τₕ_a = (τₕ,)
domains = FEDomains(τₕ_r,τₕ_a)

# Using your working syntax!
feop = LinearParamOperator(r,a,D,U,V,domains)

fe_solver = LUSolver()

# Generate Snapshots
println("Generating High-Fidelity Steady Snapshots...")
s,_ = solution_snapshots(fe_solver,feop,μₒₙ)
println("Snapshots generated. Shape: ",size(get_all_data(s)))

# ===================================================================
# 2. NEURAL OPERATOR SETUP
# ===================================================================
model = DeepONet(
  branch_layers = (1, 32, 32, 16),
  trunk_layers  = (1, 32, 32, 16),
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
println("\nStarting Offline Phase...")
neural_rb_op = reduced_operator(solver,feop,s)
println("Offline phase completed successfully!")

# ===================================================================
# 4. ONLINE PHASE (INFERENCE)
# ===================================================================
println("\nStarting Online Phase...")
r_test = realisation(D; nparams=1,sampling=:halton)

x_approx,stats = solve(solver,neural_rb_op,r_test)

predictions = get_all_data(x_approx.fe_data)
println("Online phase completed!")
println("Inference time: $(stats.time) seconds")
println("Predictions shape: $(size(predictions))")