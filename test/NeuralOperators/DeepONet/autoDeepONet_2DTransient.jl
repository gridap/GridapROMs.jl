using Gridap
using Gridap.ODEs
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.RBTransient
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs
using LinearAlgebra

println("=== AUTO-DEEPONET TRANSIENT TEST (2D HEAT EQUATION) ===")

# ===================================================================
# 1. PHYSICS & FEM SETUP
# ===================================================================
# Geometry (Extremely coarse for fast execution)
domain = (0.0, 1.0, 0.0, 1.0)
partition = (5, 5)
Ωₕ = CartesianDiscreteModel(domain, partition)
τₕ = Triangulation(Ωₕ)

# Temporal grid
θ = 1.0 # Backward Euler
dt = 0.05
t0 = 0.0
tf = 0.2
tdomain = t0:dt:tf

# Parametric quantities
pdomain = (1.0, 2.0)
D_p = ParamSpace(pdomain)
D = TransientParamSpace(pdomain, tdomain)

# Boundary condition (Homogeneous Dirichlet)
u(μ, t) = x -> 0.0
uₚₜ(μ, t) = parameterise(u, μ, t)

# Source term: f(x) = μ
f(μ, t) = x -> μ[1]
fₚₜ(μ, t) = parameterise(f, μ, t)

# Numerical integration
order = 1
dΩₕ = Measure(τₕ, 2 * order)

# Weak form
m(μ, t, du, v, dΩ) = ∫( v * du )dΩ
a(μ, t, u,  v, dΩ) = ∫( ∇(v) ⋅ ∇(u) )dΩ
r(μ, t, u,  v, dΩ) = m(μ, t, ∂t(u), v, dΩ) + a(μ, t, u, v, dΩ) - ∫( fₚₜ(μ, t) * v )dΩ

# Domains
τₕ_a = (τₕ,)
τₕ_m = (τₕ,)
τₕ_r = (τₕ,)
domains = FEDomains(τₕ_r, (τₕ_a, τₕ_m))

# FE Spaces
reffe = ReferenceFE(lagrangian, Float64, order)
V_standard = TestFESpace(Ωₕ, reffe; dirichlet_tags="boundary")

# Crucial step for ROMs: wrap the space in OrderedFESpace
V = OrderedFESpace(V_standard)
U = TransientTrialParamFESpace(V, uₚₜ)

# Operator and Solver
feop = TransientLinearParamOperator(r, (a, m), D, U, V, domains)
fe_solver = ThetaMethod(LUSolver(), dt, θ)

# ===================================================================
# 2. DATA GENERATION (TRAIN & TEST SPLIT)
# ===================================================================
println("\nGenerating Training and Testing Datasets...")

# Sample parameters
p_train = realisation(D_p; nparams=2, sampling=:uniform)
p_test  = realisation(D_p; nparams=1, sampling=:uniform)

μ_train = TransientRealisation(p_train, tdomain)
μ_test  = TransientRealisation(p_test, tdomain)

# Initial condition: u(x, 0) = 0
u₀(μ) = x -> 0.0
u₀ₚ(μ) = parameterise(u₀, μ)
uh₀ₚ_train(μ) = interpolate_everywhere(u₀ₚ(μ), U(μ, t0))
uh₀ₚ_test(μ)  = interpolate_everywhere(u₀ₚ(μ), U(μ, t0))

s_train, _ = solution_snapshots(fe_solver, feop, μ_train, uh₀ₚ_train)
s_test, _  = solution_snapshots(fe_solver, feop, μ_test,  uh₀ₚ_test)

println("Train Snapshots Shape: ", size(get_all_data(s_train)))
println("Test  Snapshots Shape: ", size(get_all_data(s_test)))

# ===================================================================
# 3. AUTO-DEEPONET SETUP
# ===================================================================
# Notice how we completely OMIT the `model` argument.
# DeepONetStrategy will use `model = AutoDeepONet()` by default,
# automatically inferring 1 branch input and 3 trunk inputs (2D space + 1 time).
strategy = DeepONetStrategy(epochs = 10)

reduction = DeepONetReduction(strategy)
solver = NeuralOpSolver(fe_solver, reduction)

# ===================================================================
# 4. TRAINING (OFFLINE PHASE)
# ===================================================================
println("\nTraining Transient DeepONet (Offline Phase)...")
neural_rb_op = reduced_operator(solver, feop, s_train)
println("Training completed successfully!")

# ===================================================================
# 5. INFERENCE (ONLINE PHASE)
# ===================================================================
println("\nEvaluating Model on Unseen Transient Test Data...")
x_approx, stats = solve(solver, neural_rb_op, μ_test)

U_true = get_all_data(s_test)
U_pred = get_all_data(x_approx.fe_data)

n_samples = size(U_true, 2)
errors = zeros(n_samples)

for i in 1:n_samples
    true_slice = U_true[:, i, :]
    pred_slice = U_pred[:, i, :]
    
    err_norm  = norm(true_slice - pred_slice)
    true_norm = norm(true_slice)
    errors[i] = err_norm / true_norm
end

mean_rel_error = sum(errors) / length(errors) * 100

println("\n=== PERFORMANCE RESULTS ===")
println("Inference Time        : $(round(stats.time, digits=5)) s (for $n_samples samples)")
println("Mean Relative L2 Error: $(round(mean_rel_error, digits=3)) %")
println("===========================")