module TransientTransferDeepONet

using Gridap
using GridapROMs
using GridapROMs.RBSteady
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.ParamODEs
using GridapROMs.RBTransient
using Lux

# Physics Setup 
L,c = 3.0,1.0
t0,dt,tf = 0.0,0.05,1.0
tdomain = t0:dt:tf

pdomain_base = (0.01,0.1) # Base variance
D_p_base = ParamSpace(pdomain_base)
D_base = TransientParamSpace(pdomain_base,tdomain) 

model = CartesianDiscreteModel((-L,L),(150,)) 
Ω,dΩ = Triangulation(model),Measure(Triangulation(model),2)

u₀(σ) = x -> (1 / √(2 * π * σ[1])) * exp(-x[1]^2 / (2 * σ[1]))
u(σ,t) = x -> (1 / √(2 * π * σ[1])) * exp(-(x[1] - c * t)^2 / (2 * σ[1]))

uₚₜ(σ,t) = parameterise(u,σ,t)
u₀ₚ(σ) = parameterise(u₀,σ)

test = OrderedFESpace(TestFESpace(model,ReferenceFE(lagrangian,Float64,1)))
trial = TransientTrialParamFESpace(test,uₚₜ)

m(σ,t,du,v) = ∫(v * du)dΩ
a(σ,t,u,v) = ∫(v * (VectorValue(c) ⋅ ∇(u)))dΩ
r(σ,t,u,v) = m(σ,t,∂t(u),v) + a(σ,t,u,v)

feop_base = TransientLinearParamOperator(r,(a,m),D_base,trial,test)
fesolver = ThetaMethod(LUSolver(),dt,0.5)

# Base Training Data
n_base = 50 # Increased base samples
σ_base = TransientRealisation(realisation(D_p_base;nparams=n_base,sampling=:halton),tdomain)
uh₀ₚ_base(σ) = interpolate_everywhere(u₀ₚ(σ),trial(σ,t0))
s_base,_ = solution_snapshots(fesolver,feop_base,σ_base,uh₀ₚ_base)

model_arch = AutoDeepONet(width=128,depth=4,activation=Lux.gelu) # High capacity

# Neural Setup (Log transform + Space-Time subsampling)
strategy_base = NeuralOpStrategy(
    model = model_arch,
    epochs = 3000,
    batch_size = 10, # Mini-batching over 50 samples
    step_x = 2,step_t = 2, # Spatio-temporal subsampling
    branch_sampler = p -> log10.(p), # The log-transform maps multi-scale variations into a manageable feature space
    lr_scheduler = ReduceLROnPlateau(patience=200,factor=0.5f0,start_lr=1e-3),
    print_every = 500
)

solver_base = NeuralOpSolver(fesolver,DeepONetReduction(strategy_base))

println("\nTraining Base Transient Model...")
pretrained_op = reduced_operator(solver_base,feop_base,s_base)

println("\nEvaluating Base Model on Base Domain...")
n_test_base = 10
σ_test_base = TransientRealisation(realisation(D_p_base;nparams=n_test_base,sampling=:halton,start=n_base+1),tdomain)
s_test_base,festats_base = solution_snapshots(fesolver,feop_base,σ_test_base,uh₀ₚ_base)
x_approx_base,rbstats_base = solve(solver_base,pretrained_op,σ_test_base)
perf_base = eval_performance(solver_base,pretrained_op,s_test_base,x_approx_base,festats_base,rbstats_base)
println(perf_base)


# Transfer Learning Phase
println("\nTransfer Learning to massive variance scales...")
pdomain_ext = (1.0,5.0) 
D_p_ext = ParamSpace(pdomain_ext)
D_ext = TransientParamSpace(pdomain_ext,tdomain)

feop_ext = TransientLinearParamOperator(r,(a,m),D_ext,trial,test)

n_ext = 30
σ_ext = TransientRealisation(realisation(D_p_ext;nparams=n_ext,sampling=:halton),tdomain)
uh₀ₚ_ext(σ) = interpolate_everywhere(u₀ₚ(σ),trial(σ,t0))
s_ext,_ = solution_snapshots(fesolver,feop_ext,σ_ext,uh₀ₚ_ext)

strategy_transfer = NeuralOpStrategy(
    model = model_arch,
    epochs = 1500,
    batch_size = 10,
    step_x = 2,step_t = 2,
    branch_sampler = p -> log10.(p),
    lr_scheduler = ReduceLROnPlateau(patience=100,start_lr=5e-4),
    print_every = 500
)
solver_transfer = NeuralOpSolver(fesolver,DeepONetReduction(strategy_transfer))

# update_stats=true forces the model to recompute normalization statistics for the shifted, radically different physical scale
transfer_op = reduced_operator(solver_transfer,feop_ext,s_ext,pretrained_op;update_stats=true)

# Evaluation on Extended Domain
println("\nTesting Models on Extended Domain...")
n_test_ext = 10
σ_test_ext = TransientRealisation(realisation(D_p_ext;nparams=n_test_ext,sampling=:halton,start=n_ext+1),tdomain)
s_test_ext,festats_ext = solution_snapshots(fesolver,feop_ext,σ_test_ext,uh₀ₚ_ext)

println("\n--- Pre-trained Model (Out-of-Distribution) ---")
x_approx_pre,rbstats_pre = solve(solver_base,pretrained_op,σ_test_ext)
perf_pre = eval_performance(solver_base,pretrained_op,s_test_ext,x_approx_pre,festats_ext,rbstats_pre)
println(perf_pre)

println("\n--- Transfer-Learned Model ---")
x_approx_transfer,rbstats_transfer = solve(solver_transfer,transfer_op,σ_test_ext)
perf_transfer = eval_performance(solver_transfer,transfer_op,s_test_ext,x_approx_transfer,festats_ext,rbstats_transfer)
println(perf_transfer)

end