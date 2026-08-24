module SteadyPoissonNOMAD

using Gridap
using GridapROMs
using GridapROMs.ParamDataStructures
using GridapROMs.ParamFESpaces
using GridapROMs.RBSteady
using GridapSolvers
using GridapSolvers.LinearSolvers

# Physics Setup (1D Parameter Space to ensure generalization on small datasets)
pdomain = (0.1,10.0) 
pspace = ParamSpace(pdomain)

domain = (0,1,0,1)
partition = (15,15)
model = CartesianDiscreteModel(domain,partition)
Ω = Triangulation(model)
dΩ = Measure(Ω,2)

ν(μ) = x -> μ[1]
νμ(μ) = parameterise(ν,μ)

f(μ) = x -> 10.0 * sin(π * x[1]) * sin(π * x[2]) # Fixed source amplitude
fμ(μ) = parameterise(f,μ)

a(μ,u,v,dΩ) = ∫( νμ(μ) * ∇(v) ⋅ ∇(u) )dΩ
l(μ,v,dΩ) = ∫( fμ(μ) * v )dΩ
res(μ,u,v,dΩ) = a(μ,u,v,dΩ) - l(μ,v,dΩ)

# OrderedFESpace is strictly required to guarantee a consistent DoF-to-coordinate mapping for Neural Operators
test_raw = TestFESpace(model,ReferenceFE(lagrangian,Float64,1);conformity=:H1,dirichlet_tags="boundary")
test = OrderedFESpace(test_raw) 

g(μ) = x -> 0.0
gμ(μ) = parameterise(g,μ)
trial = ParamTrialFESpace(test,gμ)

feop = LinearParamOperator(res,a,pspace,trial,test,FEDomains((Ω,),(Ω,)))
fesolver = LUSolver()

# Dataset Generation
n_train = 80
μ_train = realisation(pspace;nparams=n_train,sampling=:halton)
s_train,_ = solution_snapshots(fesolver,feop,μ_train)

N_dofs = num_free_dofs(test)
N_tot = N_dofs * n_train
D_phys = num_cell_dims(model)

println("\n--- Dataset Summary ---")
println("Free DoFs per sample : $N_dofs")
println("Number of samples    : $n_train")
println("Total training points: $N_tot")
println("Input physical dims  : $D_phys")
println("-----------------------\n")

# Neural Operator Setup
# Omitting explicit models/samplers triggers the defaults (AutoNOMAD, Identity branch sampling, CosineAnnealing)
reduction = NOMADReduction(;
    epochs = 2500,
    batch_size = 512,
)

neural_solver = NeuralOpSolver(fesolver,reduction)

# Offline Phase
println("[OFFLINE] Training NOMAD Operator...")
neural_rb_op = reduced_operator(neural_solver,feop,s_train)

# Online Phase
println("\n[ONLINE] Evaluating performance on unseen test data...")
n_test = 20
μ_test = realisation(pspace;nparams=n_test,sampling=:halton,start=n_train+1)

s_test,festats = solution_snapshots(fesolver,feop,μ_test)
x_approx,rbstats = solve(neural_solver,neural_rb_op,μ_test)

perf = eval_performance(neural_solver,neural_rb_op,s_test,x_approx,festats,rbstats)
println(perf)

end