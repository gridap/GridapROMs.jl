module SteadyContinualDeepONet

using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.RBSteady
using GridapSolvers
using GridapSolvers.LinearSolvers
using Lux

# Base Physics Setup (2D Parameter Space)
pdomain_base = (0.1,5.0,0.1,5.0)
pspace_base = ParamSpace(pdomain_base)

model = CartesianDiscreteModel((0,1,0,1),(25,25))
Ω,dΩ = Triangulation(model),Measure(Triangulation(model),2)

ν(μ) = x -> μ[1]
νμ(μ) = parameterise(ν,μ)
f(μ) = x -> μ[2] * sin(π * x[1]) * sin(π * x[2])
fμ(μ) = parameterise(f,μ)

a(μ,u,v,dΩ) = ∫( νμ(μ) * ∇(v) ⋅ ∇(u) )dΩ
l(μ,v,dΩ) = ∫( fμ(μ) * v )dΩ
res(μ,u,v,dΩ) = a(μ,u,v,dΩ) - l(μ,v,dΩ)

test = OrderedFESpace(TestFESpace(model,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary"))

g(μ) = x -> 0.0
gμ(μ) = parameterise(g,μ)
trial = ParamTrialFESpace(test,gμ)

feop_base = LinearParamOperator(res,a,pspace_base,trial,test,FEDomains((Ω,),(Ω,)))
fesolver = LUSolver()

n_base = 150 # Increased base samples for a 2D parameter space
s_base,_ = solution_snapshots(fesolver,feop_base,realisation(pspace_base;nparams=n_base,sampling=:halton))

# Branch Sampler Setup
# Extracts 50 sensor readings evaluated on a 1D grid to feed the Branch Net, bypassing raw parameters
x_sensors = range(0,1,length=25)

# Define the physical sensor transformations
sensor_ν_func(x,μ₁) = μ₁ * x^2
sensor_f_func(x,μ₂) = μ₂ * sin(π * x)

branch_sampler_func = (μ) -> begin
    sensors_ν = sensor_ν_func.(x_sensors,μ[1])
    sensors_f = sensor_f_func.(x_sensors,μ[2])
    return vcat(sensors_ν,sensors_f) # Output vector length: 50
end

# Explicit DeepONet initialization with matched sensor dimensionality (50 inputs)
model_arch = DeepONet(branch_layers=(50,128,128,64),trunk_layers=(2,128,128,64),activation=Lux.gelu)

strategy_base = NeuralOpStrategy(
    model = model_arch,
    epochs = 4000,
    batch_size = 25, # Mini-batching over n_samples (150/25 = 6 batches per epoch)
    step_x = 2, # Spatial subsampling: speeds up training
    branch_sampler = branch_sampler_func,
    lr_scheduler = CosineAnnealing(lr_max=1f-3,lr_min=1f-5),
    print_every = 500
)

solver_base = NeuralOpSolver(fesolver,DeepONetReduction(strategy_base))

println("\nTraining Base Model...")
pretrained_op = reduced_operator(solver_base,feop_base,s_base)

println("\nEvaluating Base Model on Base Domain...")
n_test_base = 20
μ_test_base = realisation(pspace_base;nparams=n_test_base,sampling=:halton,start=n_base+1)
s_test_base,festats_base = solution_snapshots(fesolver,feop_base,μ_test_base)
x_approx_base,rbstats_base = solve(solver_base,pretrained_op,μ_test_base)
perf_base = eval_performance(solver_base,pretrained_op,s_test_base,x_approx_base,festats_base,rbstats_base)
println(perf_base)


# Continual Learning Phase
println("\nContinual Learning on Extended Domain...")
pdomain_ext = (5.0,10.0,5.0,10.0)
pspace_ext = ParamSpace(pdomain_ext)
feop_ext = LinearParamOperator(res,a,pspace_ext,trial,test,FEDomains((Ω,),(Ω,)))

n_ext = 60
s_ext,_ = solution_snapshots(fesolver,feop_ext,realisation(pspace_ext;nparams=n_ext,sampling=:halton))

strategy_ft = NeuralOpStrategy(
    model = model_arch,
    epochs = 2000,
    batch_size = 15, # Mini-batching over 60 samples
    step_x = 2,
    branch_sampler = branch_sampler_func,
    lr_scheduler = CosineAnnealing(lr_max=1f-4,lr_min=1f-6),
    print_every = 500
)
solver_ft = NeuralOpSolver(fesolver,DeepONetReduction(strategy_ft))

# update_stats=false enforces physical consistency: the model inherits the normalization Z-scores (μ, σ) of the pre-trained domain
finetuned_op = reduced_operator(solver_ft,feop_ext,s_ext,pretrained_op;update_stats=false)

# Evaluation on Extended Domain
println("\nTesting Models on Extended Domain...")
n_test_ext = 20
μ_test_ext = realisation(pspace_ext;nparams=n_test_ext,sampling=:halton,start=n_ext+1)
s_test_ext,festats_ext = solution_snapshots(fesolver,feop_ext,μ_test_ext)

println("\n--- Pre-trained Model (Out-of-Distribution) ---")
x_approx_pre,rbstats_pre = solve(solver_base,pretrained_op,μ_test_ext)
perf_pre = eval_performance(solver_base,pretrained_op,s_test_ext,x_approx_pre,festats_ext,rbstats_pre)
println(perf_pre)

println("\n--- Fine-Tuned Model ---")
x_approx_ft,rbstats_ft = solve(solver_ft,finetuned_op,μ_test_ext)
perf_ft = eval_performance(solver_ft,finetuned_op,s_test_ext,x_approx_ft,festats_ext,rbstats_ft)
println(perf_ft)

end